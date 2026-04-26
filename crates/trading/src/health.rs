//! Production-robustness layer: self-diagnostics + auto-repair.
//!
//! This module exists because real production trading systems run for
//! days/weeks and accumulate failure modes that no unit test catches:
//!
//! - **NaN/Inf creep**: a single bad indicator value or division can
//!   poison every downstream calculation, silently. Once NaN gets into
//!   a confidence value or position size, every subsequent prediction
//!   is corrupt.
//! - **Buffer saturation**: history VecDeques and PredictionLog have
//!   bounded capacity, but if the bounds aren't enforced *every push*,
//!   they leak.
//! - **Stale calibration**: the calibration buckets accumulate over
//!   time. If the brain's behavior changes (e.g. recent training
//!   shifted patterns), old bucket data may no longer reflect current
//!   accuracy. Calibration becomes wrong by inertia.
//! - **Anomaly drift**: predictive-coding error has no upper bound;
//!   over many ticks it can grow large enough that the anomaly score
//!   sigmoid pegs at 1.0 for everything, making the signal useless.
//! - **History bias**: if recent training has been one-sided (say all
//!   Up outcomes during a bull run), the empirical distribution skews
//!   and the brain becomes systematically wrong on the eventual
//!   reversal.
//! - **Score-by-key staleness**: PredictionLog.score_by_key() walks
//!   from the back, but if predictions accumulate without outcomes,
//!   the unscored backlog grows.
//!
//! `health_check()` looks for these. `auto_repair()` fixes them when
//! safe to do so. `/api/trading/health` exposes the result so a human
//! or external monitor can act before catastrophe.

use serde::{Deserialize, Serialize};

/// Severity ladder for diagnostic findings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {
    /// Informational — nothing wrong, just FYI.
    Info,
    /// Concerning trend, no action required yet.
    Warning,
    /// Likely bug or imminent failure — investigate.
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthFinding {
    pub severity: Severity,
    pub category: String,
    pub message: String,
    /// Did auto_repair fix this? None = couldn't, true/false = tried.
    pub repaired: Option<bool>,
}

/// Aggregated health report. Designed for automated polling: a monitor
/// can check `worst_severity()` and alert; a human can read `findings`
/// for detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Total predictions ever made (lifetime).
    pub lifetime_predictions: u64,
    /// Trade-history buffer size (current).
    pub history_size: usize,
    /// Trade-history capacity (max).
    pub history_capacity: usize,
    /// Prediction-log size (current).
    pub prediction_log_size: usize,
    /// Prediction-log capacity (max).
    pub prediction_log_capacity: usize,
    /// Number of unscored predictions in the log. Excessive backlog
    /// means analyze() is being called without follow-up
    /// train_on_outcome().
    pub unscored_predictions: usize,
    /// Findings: anything anomalous about internal state.
    pub findings: Vec<HealthFinding>,
}

impl HealthReport {
    /// Worst severity across all findings. Use for alerting thresholds.
    pub fn worst_severity(&self) -> Severity {
        let mut worst = Severity::Info;
        for f in &self.findings {
            match (worst, f.severity) {
                (_, Severity::Critical) => return Severity::Critical,
                (Severity::Info, Severity::Warning) => worst = Severity::Warning,
                _ => {}
            }
        }
        worst
    }

    pub fn has_critical(&self) -> bool {
        self.findings.iter().any(|f| f.severity == Severity::Critical)
    }
}

/// Sanity-check a float. Returns Some(reason) if it should be repaired,
/// None if it's fine.
pub fn float_issue(name: &str, value: f32) -> Option<String> {
    if value.is_nan() {
        Some(format!("{} is NaN", name))
    } else if value.is_infinite() {
        Some(format!("{} is Inf", name))
    } else if value.abs() > 1e6 {
        Some(format!("{} = {:.3e} is implausibly large", name, value))
    } else {
        None
    }
}

/// Sanity-check a f64 (for MarketState fields like price/volume).
pub fn float_issue_f64(name: &str, value: f64) -> Option<String> {
    if value.is_nan() {
        Some(format!("{} is NaN", name))
    } else if value.is_infinite() {
        Some(format!("{} is Inf", name))
    } else {
        None
    }
}

/// Validate and sanitize a MarketState. Returns the cleaned state and a
/// list of issues found (and silently fixed). Returns None if the state
/// is too corrupted to repair (e.g. price is NaN AND volume is NaN —
/// no signal left).
///
/// The defensive principle: NEVER let bad input contaminate the brain's
/// internal state. Sanitize at the boundary, log the issues, proceed
/// with safe values.
pub fn validate_and_clean_state(
    state: &crate::MarketState,
) -> Result<(crate::MarketState, Vec<String>), String> {
    let mut issues: Vec<String> = Vec::new();
    let mut clean = state.clone();

    // Price: NaN/Inf is unrecoverable (we can't make up a price).
    if let Some(msg) = float_issue_f64("price", state.price) {
        return Err(format!("price unrecoverable: {}", msg));
    }
    if state.price <= 0.0 {
        issues.push("price <= 0, clamped to 1e-6".to_string());
        clean.price = 1e-6;
    }

    // Volume: clamp to non-negative.
    if let Some(msg) = float_issue_f64("volume", state.volume) {
        issues.push(format!("volume: {}, set to 0", msg));
        clean.volume = 0.0;
    } else if state.volume < 0.0 {
        issues.push("volume < 0, clamped to 0".to_string());
        clean.volume = 0.0;
    }

    // Returns and volatility: clamp NaN/Inf to 0.
    if let Some(msg) = float_issue_f64("recent_return", state.recent_return) {
        issues.push(format!("recent_return: {}, set to 0", msg));
        clean.recent_return = 0.0;
    }
    if let Some(msg) = float_issue_f64("volatility", state.volatility) {
        issues.push(format!("volatility: {}, set to 0", msg));
        clean.volatility = 0.0;
    } else if state.volatility < 0.0 {
        clean.volatility = state.volatility.abs();
    }

    // Indicators: drop any with NaN/Inf values rather than poisoning.
    let original_n = state.indicators.len();
    clean.indicators.retain(|(name, value)|
        float_issue_f64(name, *value).is_none());
    if clean.indicators.len() < original_n {
        issues.push(format!(
            "dropped {} indicators with NaN/Inf",
            original_n - clean.indicators.len()));
    }

    // News items: drop any with NaN sentiment.
    let original_news = state.news.len();
    clean.news.retain(|n| float_issue_f64("sentiment", n.sentiment).is_none()
        && float_issue_f64("age_hours", n.age_hours).is_none());
    if clean.news.len() < original_news {
        issues.push(format!(
            "dropped {} news items with bad floats",
            original_news - clean.news.len()));
    }

    Ok((clean, issues))
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use crate::{MarketState, NewsItem, TimeContext};

    fn s() -> MarketState {
        MarketState {
            price: 100.0, volume: 1000.0,
            recent_return: 0.01, volatility: 0.02,
            indicators: vec![("rsi".into(), 50.0)],
            news: vec![NewsItem {
                source: "FED".into(), headline: "ok".into(),
                sentiment: 0.3, age_hours: 1.0,
            }],
            timestamp: Some(TimeContext { hour_utc: 12, day_of_week: 2 }),
        }
    }

    #[test]
    fn test_clean_passes_valid_state() {
        let (cleaned, issues) = validate_and_clean_state(&s()).unwrap();
        assert!(issues.is_empty(), "valid state should produce no issues, got {:?}", issues);
        assert_eq!(cleaned.price, 100.0);
    }

    #[test]
    fn test_clean_rejects_nan_price() {
        let mut bad = s();
        bad.price = f64::NAN;
        assert!(validate_and_clean_state(&bad).is_err(),
            "NaN price must be unrecoverable error");
    }

    #[test]
    fn test_clean_zeros_nan_return() {
        let mut bad = s();
        bad.recent_return = f64::NAN;
        let (cleaned, issues) = validate_and_clean_state(&bad).unwrap();
        assert_eq!(cleaned.recent_return, 0.0);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_clean_drops_bad_indicators() {
        let mut bad = s();
        bad.indicators.push(("bad".into(), f64::NAN));
        let (cleaned, issues) = validate_and_clean_state(&bad).unwrap();
        assert_eq!(cleaned.indicators.len(), 1);
        assert!(!issues.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        let mut report = HealthReport {
            lifetime_predictions: 0,
            history_size: 0, history_capacity: 0,
            prediction_log_size: 0, prediction_log_capacity: 0,
            unscored_predictions: 0,
            findings: vec![
                HealthFinding {
                    severity: Severity::Warning,
                    category: "test".into(),
                    message: "warn".into(),
                    repaired: None,
                },
                HealthFinding {
                    severity: Severity::Info,
                    category: "test".into(),
                    message: "info".into(),
                    repaired: None,
                },
            ],
        };
        assert_eq!(report.worst_severity(), Severity::Warning);
        report.findings.push(HealthFinding {
            severity: Severity::Critical,
            category: "test".into(),
            message: "boom".into(),
            repaired: None,
        });
        assert_eq!(report.worst_severity(), Severity::Critical);
        assert!(report.has_critical());
    }

    #[test]
    fn test_float_issue_detects_nan_and_inf() {
        assert!(float_issue("x", f32::NAN).is_some());
        assert!(float_issue("x", f32::INFINITY).is_some());
        assert!(float_issue("x", f32::NEG_INFINITY).is_some());
        assert!(float_issue("x", 1e10).is_some());
        assert!(float_issue("x", 0.5).is_none());
        assert!(float_issue("x", -0.5).is_none());
        assert!(float_issue("x", 0.0).is_none());
    }
}
