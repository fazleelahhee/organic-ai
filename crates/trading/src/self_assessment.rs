//! Self-awareness layer: the trading reasoner tracks its own predictions
//! against realized outcomes and exposes structured introspection so a
//! trader (or downstream risk-management system) can see exactly what
//! the brain knows about its own track record.
//!
//! What "self-aware" means concretely here:
//!
//! 1. **Prediction logging**. Every analyze() pushes a PredictionRecord
//!    onto an internal log. The record has the encoded query key, the
//!    predicted direction, confidence, anomaly score, regime tag, and a
//!    monotonic ID.
//!
//! 2. **Retroactive scoring**. When train_on_outcome() is called for a
//!    state similar to a logged prediction, the prediction is matched
//!    and scored — was it right? close on magnitude? — and the record
//!    is finalized.
//!
//! 3. **Aggregate accuracy stats**. Overall hit rate, hit rate by
//!    regime tag, hit rate by news source, hit rate by confidence
//!    bucket (calibration curve). All updated incrementally.
//!
//! 4. **Calibration tracking**. The reasoner builds a calibration curve
//!    — when it said "0.7 confidence", how often was that actually
//!    right? — and exposes it. A well-calibrated 0.7 should be right
//!    ~70% of the time; if it's right 50%, the model is overconfident
//!    and downstream consumers should discount.
//!
//! 5. **Drift detection**. Rolling-window accuracy compared to
//!    historical baseline. Drift flag goes high when recent accuracy
//!    falls below long-term by a configurable margin. Production-
//!    critical: a brain that's silently wrong is worse than one that
//!    admits it.
//!
//! What this does NOT do:
//! - Doesn't automatically retrain or modify weights based on track
//!   record. That kind of meta-learning is a research project; this
//!   layer instead surfaces the data so a human (or higher-level
//!   automation) can decide.
//! - Doesn't replace good monitoring/alerting infra in production —
//!   complement it.

use crate::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single logged prediction. Created by `analyze()`, finalized by
/// `report_outcome()` when the realized direction is known.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// Monotonic ID assigned at prediction time.
    pub id: u64,
    /// The encoded state key (for matching to the outcome later).
    pub state_key: String,
    /// Regime tag extracted from the state key (for per-regime stats).
    pub regime: String,
    /// News source tags extracted from the state key (for per-source
    /// stats). Empty if the state had no news.
    pub news_sources: Vec<String>,
    /// What the brain predicted.
    pub predicted: Direction,
    /// Brain's reported confidence.
    pub confidence: f32,
    /// Brain's reported anomaly score.
    pub anomaly_score: f32,
    /// Set when the outcome is reported. None until then.
    pub actual: Option<Direction>,
    /// True if predicted == actual. None until reported.
    pub hit: Option<bool>,
}

/// Aggregated self-assessment over the prediction log. Returned by
/// `TradingBrain::self_assessment()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessment {
    /// Total predictions made (logged).
    pub total_predictions: u64,
    /// Total predictions that have been scored against an outcome.
    pub scored_predictions: u64,
    /// Overall hit rate over scored predictions.
    pub overall_accuracy: f32,
    /// Hit rate per regime tag (regime_euphoric, regime_oversold, etc.).
    pub accuracy_by_regime: HashMap<String, RegimeStats>,
    /// Hit rate per news source (FED, SEC, TWITTER, etc.).
    pub accuracy_by_source: HashMap<String, RegimeStats>,
    /// Calibration curve: bucket index 0..=9 corresponds to confidence
    /// in [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]. For each bucket,
    /// reports observed hit rate. A perfectly-calibrated model has
    /// observed_hit_rate ≈ bucket_midpoint.
    pub calibration: Vec<CalibrationBucket>,
    /// Recent-window hit rate (last `recent_window_size` scored).
    pub recent_accuracy: f32,
    /// True when recent_accuracy < overall_accuracy by drift_threshold.
    pub drift_detected: bool,
    /// Window size used for the recent_accuracy calculation.
    pub recent_window_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeStats {
    pub n_predictions: u64,
    pub n_hits: u64,
    pub hit_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBucket {
    /// Lower bound of the confidence bucket (0.0, 0.1, ..., 0.9).
    pub conf_lower: f32,
    /// Upper bound (exclusive, except top bucket which is inclusive).
    pub conf_upper: f32,
    /// Number of predictions whose confidence fell in this bucket.
    pub n: u64,
    /// Observed hit rate in this bucket.
    pub hit_rate: f32,
    /// Calibration error: |hit_rate - bucket_midpoint|. Lower = better
    /// calibrated. >0.15 in any well-populated bucket is concerning.
    pub calibration_error: f32,
}

/// The internal performance log. Owned by TradingBrain. Bounded so it
/// can run indefinitely in production without unbounded memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionLog {
    pub records: std::collections::VecDeque<PredictionRecord>,
    pub next_id: u64,
    pub max_records: usize,
    /// Window size for the rolling-recent accuracy calculation.
    pub recent_window: usize,
    /// Drift threshold: if recent accuracy is below overall by this
    /// margin, drift_detected fires. 0.10 = "10 percentage points
    /// degradation triggers an alert."
    pub drift_threshold: f32,
}

impl Default for PredictionLog {
    fn default() -> Self {
        Self {
            records: std::collections::VecDeque::new(),
            next_id: 1,
            max_records: 10_000,
            recent_window: 50,
            drift_threshold: 0.10,
        }
    }
}

impl PredictionLog {
    pub fn new() -> Self { Self::default() }

    /// Push a fresh prediction. Returns the assigned ID — callers
    /// typically include it in the Analysis they return so consumers
    /// can correlate later outcomes back to this record.
    pub fn record(&mut self, mut record: PredictionRecord) -> u64 {
        record.id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.records.push_back(record);
        while self.records.len() > self.max_records {
            self.records.pop_front();
        }
        self.next_id - 1
    }

    /// Find the most recent unscored record matching `state_key` and
    /// finalize it with the given actual direction. Returns true if a
    /// match was found and scored. Used by train_on_outcome to close
    /// the loop without explicit IDs.
    pub fn score_by_key(&mut self, state_key: &str, actual: Direction) -> bool {
        for r in self.records.iter_mut().rev() {
            if r.actual.is_none() && r.state_key == state_key {
                r.actual = Some(actual);
                r.hit = Some(r.predicted == actual);
                return true;
            }
        }
        false
    }

    /// Compute the full SelfAssessment over the log.
    pub fn assess(&self) -> SelfAssessment {
        let scored: Vec<&PredictionRecord> = self.records.iter()
            .filter(|r| r.hit.is_some()).collect();
        let total_predictions = self.records.len() as u64;
        let scored_predictions = scored.len() as u64;

        let n_hits = scored.iter().filter(|r| r.hit == Some(true)).count();
        let overall_accuracy = if scored_predictions == 0 { 0.0 }
            else { n_hits as f32 / scored_predictions as f32 };

        // Per regime.
        let mut by_regime: HashMap<String, (u64, u64)> = HashMap::new();
        for r in &scored {
            let entry = by_regime.entry(r.regime.clone()).or_insert((0, 0));
            entry.0 += 1;
            if r.hit == Some(true) { entry.1 += 1; }
        }
        let accuracy_by_regime: HashMap<String, RegimeStats> = by_regime.into_iter()
            .map(|(k, (n, h))| {
                let rate = if n == 0 { 0.0 } else { h as f32 / n as f32 };
                (k, RegimeStats { n_predictions: n, n_hits: h, hit_rate: rate })
            }).collect();

        // Per source. A prediction can have multiple sources; each
        // contributes independently.
        let mut by_source: HashMap<String, (u64, u64)> = HashMap::new();
        for r in &scored {
            for src in &r.news_sources {
                let entry = by_source.entry(src.clone()).or_insert((0, 0));
                entry.0 += 1;
                if r.hit == Some(true) { entry.1 += 1; }
            }
        }
        let accuracy_by_source: HashMap<String, RegimeStats> = by_source.into_iter()
            .map(|(k, (n, h))| {
                let rate = if n == 0 { 0.0 } else { h as f32 / n as f32 };
                (k, RegimeStats { n_predictions: n, n_hits: h, hit_rate: rate })
            }).collect();

        // Calibration curve.
        let mut calibration: Vec<CalibrationBucket> = (0..10).map(|i| {
            let lo = i as f32 / 10.0;
            let hi = (i + 1) as f32 / 10.0;
            let in_bucket: Vec<&&PredictionRecord> = scored.iter()
                .filter(|r| {
                    let c = r.confidence;
                    if i == 9 { c >= lo && c <= hi } else { c >= lo && c < hi }
                }).collect();
            let n = in_bucket.len() as u64;
            let hits = in_bucket.iter().filter(|r| r.hit == Some(true)).count();
            let hit_rate = if n == 0 { 0.0 } else { hits as f32 / n as f32 };
            let mid = (lo + hi) / 2.0;
            let calibration_error = (hit_rate - mid).abs();
            CalibrationBucket {
                conf_lower: lo, conf_upper: hi, n, hit_rate, calibration_error,
            }
        }).collect();
        // Suppress error reporting for empty buckets.
        for b in &mut calibration {
            if b.n == 0 { b.calibration_error = 0.0; }
        }

        // Recent accuracy: last `recent_window` scored predictions.
        let recent: Vec<&&PredictionRecord> = scored.iter().rev()
            .take(self.recent_window).collect();
        let recent_n = recent.len() as f32;
        let recent_hits = recent.iter().filter(|r| r.hit == Some(true)).count() as f32;
        let recent_accuracy = if recent_n == 0.0 { 0.0 } else { recent_hits / recent_n };

        let drift_detected = scored_predictions >= self.recent_window as u64
            && (overall_accuracy - recent_accuracy) >= self.drift_threshold;

        SelfAssessment {
            total_predictions,
            scored_predictions,
            overall_accuracy,
            accuracy_by_regime,
            accuracy_by_source,
            calibration,
            recent_accuracy,
            drift_detected,
            recent_window_size: self.recent_window,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: u64, state_key: &str, regime: &str, predicted: Direction, conf: f32) -> PredictionRecord {
        PredictionRecord {
            id,
            state_key: state_key.to_string(),
            regime: regime.to_string(),
            news_sources: Vec::new(),
            predicted,
            confidence: conf,
            anomaly_score: 0.5,
            actual: None,
            hit: None,
        }
    }

    #[test]
    fn test_log_assigns_monotonic_ids() {
        let mut log = PredictionLog::new();
        let id1 = log.record(rec(0, "k1", "regime_normal", Direction::Up, 0.5));
        let id2 = log.record(rec(0, "k2", "regime_normal", Direction::Down, 0.5));
        assert!(id2 > id1);
    }

    #[test]
    fn test_score_by_key_finds_unscored() {
        let mut log = PredictionLog::new();
        log.record(rec(0, "k1", "regime_normal", Direction::Up, 0.5));
        assert!(log.score_by_key("k1", Direction::Up));
        let assessed = log.assess();
        assert_eq!(assessed.scored_predictions, 1);
        assert_eq!(assessed.overall_accuracy, 1.0);
    }

    #[test]
    fn test_assess_overall_accuracy() {
        let mut log = PredictionLog::new();
        log.record(rec(0, "a", "regime_normal", Direction::Up, 0.6));
        log.record(rec(0, "b", "regime_normal", Direction::Up, 0.6));
        log.record(rec(0, "c", "regime_normal", Direction::Down, 0.7));
        log.score_by_key("a", Direction::Up);
        log.score_by_key("b", Direction::Down);
        log.score_by_key("c", Direction::Down);

        let a = log.assess();
        assert_eq!(a.scored_predictions, 3);
        assert!((a.overall_accuracy - 2.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_per_regime_stats() {
        let mut log = PredictionLog::new();
        log.record(rec(0, "a", "regime_euphoric", Direction::Up, 0.6));
        log.record(rec(0, "b", "regime_euphoric", Direction::Up, 0.6));
        log.record(rec(0, "c", "regime_oversold", Direction::Down, 0.7));
        log.score_by_key("a", Direction::Up);   // hit
        log.score_by_key("b", Direction::Down); // miss
        log.score_by_key("c", Direction::Down); // hit

        let a = log.assess();
        let euphoric = a.accuracy_by_regime.get("regime_euphoric").unwrap();
        assert_eq!(euphoric.n_predictions, 2);
        assert_eq!(euphoric.n_hits, 1);
        let oversold = a.accuracy_by_regime.get("regime_oversold").unwrap();
        assert_eq!(oversold.n_predictions, 1);
        assert_eq!(oversold.n_hits, 1);
    }

    #[test]
    fn test_calibration_buckets() {
        let mut log = PredictionLog::new();
        // 10 predictions all at conf 0.85 (bucket 8: [0.8, 0.9)).
        // 8 hit, 2 miss → observed hit rate 0.8 in that bucket.
        for i in 0..10 {
            let mut r = rec(0, &format!("k{}", i), "regime_normal", Direction::Up, 0.85);
            log.record(r.clone());
            log.score_by_key(&r.state_key, if i < 8 { Direction::Up } else { Direction::Down });
        }
        let a = log.assess();
        let bucket8 = &a.calibration[8];
        assert_eq!(bucket8.n, 10);
        assert!((bucket8.hit_rate - 0.8).abs() < 0.01);
        // Bucket midpoint is 0.85, hit rate 0.8 → error 0.05.
        assert!((bucket8.calibration_error - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_drift_detection() {
        let mut log = PredictionLog::new();
        log.recent_window = 5;
        log.drift_threshold = 0.10;

        // 50 predictions: first 45 mostly hits (~80%), last 5 all misses.
        // Recent accuracy = 0/5 = 0, overall ≈ 36/50 = 0.72.
        // Drift = 0.72 - 0 = 0.72 > threshold 0.10 → detected.
        for i in 0..45 {
            log.record(rec(0, &format!("k{}", i), "regime_normal", Direction::Up, 0.6));
            let actual = if i % 5 == 0 { Direction::Down } else { Direction::Up };
            log.score_by_key(&format!("k{}", i), actual);
        }
        for i in 45..50 {
            log.record(rec(0, &format!("k{}", i), "regime_normal", Direction::Up, 0.6));
            log.score_by_key(&format!("k{}", i), Direction::Down);
        }
        let a = log.assess();
        assert!(a.drift_detected, "drift should be detected with recent miss streak");
    }
}
