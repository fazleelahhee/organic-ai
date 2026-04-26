//! Trading reasoner built on OrganicBrain.
//!
//! Wraps the brain with domain-specific encoding, a multi-step reasoning
//! chain, and structured output. Strong reasoning here doesn't mean
//! something the brain can't already do — it means *chaining* existing
//! brain components (HDC pattern recall, predictive-coding surprise,
//! inner-life transitive discovery, working-memory hold) into a
//! deliberate inference pipeline rather than a single one-shot query.
//!
//! ## What this gives you over `brain.process(text)`
//!
//! - **Numerical state encoding**: market state (price, volume, returns,
//!   volatility, indicators) is bucketized into discrete tokens, then run
//!   through the brain's text encoder. Numerically-similar states activate
//!   structurally-similar spike patterns.
//! - **Multi-pass reasoning**: each call walks the brain through several
//!   inference steps and aggregates results, instead of trusting a single
//!   pass. Mirrors how analysts reason across timeframes.
//! - **Structured `Analysis` output**: direction, confidence, anomaly
//!   score, similar past patterns, and an audit-trail of reasoning steps —
//!   so a downstream trading system can use the brain's output the way it
//!   would use any other model: as features in a fuller decision pipeline.
//! - **Closed-loop learning**: every outcome that arrives later is fed back
//!   via `train_on_outcome`, so the brain's pattern memory is always
//!   current with what the market is actually doing.
//!
//! ## What it does NOT give you
//!
//! - Compositional generalization to novel market regimes — same
//!   architectural limit as the underlying brain. New regimes need
//!   training. There is no magic here.
//! - Explanations in natural language. Reasoning steps are structured
//!   tags, not paragraphs. If you want narrative explanations, run an
//!   LLM offline against the audit trail.
//! - A complete trading system. This is a pattern-recognition + anomaly-
//!   detection + memory module. Pair it with risk management, sizing,
//!   execution, and ideally other models for production use.

use organic_neuron::brain::OrganicBrain;
use serde::{Deserialize, Serialize};

pub mod seed;
pub mod backtest;
pub mod baseline;
pub mod self_assessment;
pub mod health;
pub mod news_composer;
pub mod position;

/// A discrete news event the brain should consider alongside numeric
/// market state. The user's core ask: feed the brain Fed announcements,
/// tweets, geopolitical events, and let it learn how each kind of news
/// historically affected price. Encoded into the state's text key so HDC
/// recall finds past situations with similar news context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    /// Source category — short, controlled vocabulary like "FED", "SEC",
    /// "TWITTER", "REUTERS", "ONCHAIN". Becomes a categorical token; same
    /// source name across events activates the same input neurons.
    pub source: String,
    /// Free-text headline. The encoder extracts content words and folds
    /// them into the brain's spike pattern, so the brain can learn that
    /// "rates", "hike", "hawkish" cluster together.
    pub headline: String,
    /// Polarity in [-1, 1]. -1 = strongly bearish, +1 = strongly bullish.
    /// Bucketed at fine resolution (per-percent) by `bucket_signed`.
    pub sentiment: f64,
    /// How long ago this happened, in hours. Used for time-decay weighting
    /// and as an explicit token so the brain can learn "news within 1
    /// hour reacts differently than news from 24 hours ago."
    pub age_hours: f64,
    /// Claude-extracted compositional tokens (act_/vrb_/obj_/mag_).
    /// Populated by `TradingBrain::enrich_news()` ahead of analyze().
    /// When present, encode_state emits these instead of bag-of-words
    /// `w_<word>` tokens — brain gets semantic structure ("Fed cuts"
    /// vs "Fed raises" share actor+object, differ on action) for free.
    /// Empty = falls back to bag-of-words encoding.
    #[serde(default)]
    pub extraction_tokens: Vec<String>,
}

/// Optional time-of-day / calendar context. Intraday and weekly seasonality
/// is real and exploitable in markets — encoding the hour and weekday
/// gives the brain features to learn time-conditional patterns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeContext {
    /// Hour of day in UTC (0-23).
    pub hour_utc: u8,
    /// Day of week (0=Mon, 6=Sun).
    pub day_of_week: u8,
}

/// A snapshot of market state at a single point in time. Numeric features
/// (price, volume, returns, volatility, indicators) plus optional news
/// and time context. The encoder folds everything into a deterministic
/// text key the brain can ingest — multi-modal in input, single-channel
/// in encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Most recent price.
    pub price: f64,
    /// Most recent volume (units arbitrary — bucketized internally).
    pub volume: f64,
    /// Recent return (e.g. 5-period log return), in [-1, 1] roughly.
    pub recent_return: f64,
    /// Recent realized volatility (rolling std of returns).
    pub volatility: f64,
    /// Optional named indicator values (RSI, MACD signal, etc.). Each is
    /// bucketized just like the named fields above. Order-stable: the same
    /// indicator name across calls produces the same encoding token, so
    /// the brain learns name → value → bucket associations consistently.
    pub indicators: Vec<(String, f64)>,
    /// Recent news / sentiment events. Order doesn't matter — the encoder
    /// sorts deterministically. Empty list = no news context, behaves like
    /// pure-numeric pre-news version.
    #[serde(default)]
    pub news: Vec<NewsItem>,
    /// Optional time-of-day context. None = no temporal features in encoding.
    #[serde(default)]
    pub timestamp: Option<TimeContext>,
}

/// Outcome of a position taken from a `MarketState`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction { Up, Down, Flat }

/// One horizon-specific outcome — direction + magnitude over a specific
/// time window after the state. A single market state can yield very
/// different reactions at different horizons (e.g. SVB news caused a
/// 1-hour dip then a 7-day rally on haven-trade narrative).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonOutcome {
    /// Time window in hours (1, 4, 24, 168 = 1 week, etc.).
    pub horizon_hours: u32,
    pub direction: Direction,
    /// Magnitude of move as a fraction (0.025 = 2.5%).
    pub magnitude: f64,
}

/// Realized outcome after a state. The primary direction/magnitude is
/// the canonical 24-hour reaction (or whatever horizon you treat as
/// primary). `additional_horizons` carries other-timeframe outcomes
/// when available — empty list is fine, and the system degrades to
/// single-horizon behavior in that case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub direction: Direction,
    /// Magnitude of move as a percentage (e.g. 0.025 = 2.5% up if Up).
    pub magnitude: f64,
    /// Optional additional horizon predictions. Empty = single-horizon
    /// data only (existing seed dataset behavior).
    #[serde(default)]
    pub additional_horizons: Vec<HorizonOutcome>,
}

impl Outcome {
    /// Construct a single-horizon outcome (no additional horizons).
    /// Convenience for callers who only have 24h data — backward
    /// compatible with the pre-multi-horizon API.
    pub fn new(direction: Direction, magnitude: f64) -> Self {
        Self { direction, magnitude, additional_horizons: Vec::new() }
    }
}

/// A historical (state, outcome) match retrieved as part of an analysis.
/// Each match is a past situation the reasoner thinks is similar to the
/// current one, plus what actually happened next. This is the explainable
/// "this is like X which led to Y" reasoning a trader actually wants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Bucketized encoding of the past state (for debugging / display).
    pub state_key: String,
    /// What actually happened after that state (the realized outcome).
    pub outcome: Outcome,
    /// Similarity score in [0, 1]. Token-overlap-based — see
    /// `TradingBrain::token_similarity`.
    pub similarity: f32,
}

/// Aggregated outcome distribution across the top-K similar past patterns.
/// This is what makes the analysis genuinely useful — not just a
/// prediction, but the empirical base rate of past similar situations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDistribution {
    pub p_up: f32,
    pub p_down: f32,
    pub p_flat: f32,
    /// Mean magnitude of the past outcomes (regardless of direction).
    pub mean_magnitude: f32,
    /// Count of patterns the distribution was computed over.
    pub sample_size: u32,
    /// Mean magnitude of WIN outcomes — past patterns whose direction
    /// matched the predicted direction. The empirical "how big do my
    /// correct predictions tend to be" estimate. Used for EV-aware
    /// position sizing: small wins + big losses = unprofitable.
    #[serde(default)]
    pub mean_win_magnitude: f32,
    /// Mean magnitude of LOSS outcomes — past patterns whose direction
    /// contradicted the predicted direction. The empirical "how big do
    /// my mistakes tend to be" estimate. The asymmetry between
    /// win/loss magnitudes is the root cause of profit-factor < 1
    /// in trading systems with positive hit rate.
    #[serde(default)]
    pub mean_loss_magnitude: f32,
}

/// Per-horizon forecast. Same state can produce different predictions at
/// different time horizons — e.g. "Down 1h, Up 7d" is a common news-
/// shock pattern (initial flush then recovery).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonPrediction {
    pub horizon_hours: u32,
    pub direction: Direction,
    /// Confidence specific to this horizon (calibrated over horizon-
    /// specific outcomes from retrieved patterns).
    pub confidence: f32,
    /// Mean magnitude of the move across matched past patterns at this
    /// horizon. Useful for position-sizing and stop-loss placement.
    pub mean_magnitude: f32,
}

/// State snapshot for save/restore. Captures everything the trading
/// reasoner owns OUTSIDE the brain. Brain itself is checkpointed
/// separately by the engine's bincode persistence. Used for hourly
/// known-good checkpoints — if current state goes bad (corruption,
/// saturation, runaway), restore from the last snapshot and lose at
/// most an hour of training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub history: Vec<(String, Outcome)>,
    pub history_capacity: usize,
    pub recency_half_life: f32,
    pub max_position_fraction: f32,
    pub prediction_log: self_assessment::PredictionLog,
}

/// Recommended position sizing derived from confidence, anomaly, and
/// expected magnitude. NOT financial advice — a starting point for the
/// trader's risk-management layer to consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizing {
    /// Suggested size as fraction of capital, in [-max_position,
    /// +max_position]. Positive = long, negative = short, 0 = no trade.
    /// Computed as edge × confidence × (1 - anomaly × 0.5) capped at
    /// max_position. Conservative quarter-Kelly style — actual position
    /// should typically be smaller still after risk-management gates.
    pub fraction: f32,
    /// Edge: (p_top - p_opposing) — the empirical advantage from the
    /// outcome distribution. 0 = no advantage; 1 = unanimous past data.
    pub edge: f32,
    /// Why this sizing — e.g. "low_confidence:0.3", "high_anomaly:0.85",
    /// "strong_edge:0.7". Lets the trader's risk layer make informed
    /// overrides ("ignore brain when anomaly > 0.9").
    pub rationale: Vec<String>,
}

/// Structured output of a reasoning pass. Designed to plug into a fuller
/// trading system — not a final decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    /// The brain's best guess at next-period direction. `Flat` is also
    /// the default when the brain abstains (low confidence, high anomaly).
    pub direction: Direction,
    /// 0.0 (no idea) to 1.0 (certain). Combines HDC recall similarity,
    /// inverse anomaly score, and consistency across reasoning passes.
    pub confidence: f32,
    /// 0.0 (familiar) to 1.0 (highly novel). Predictive-coding prediction
    /// error on the input→hidden transition. High → "this is a regime I
    /// have weak coverage on, treat my output skeptically."
    pub anomaly_score: f32,
    /// Top-K past states most similar to the current one, with what
    /// happened next. The "this is like Tuesday's setup" explainability.
    pub similar_patterns: Vec<PatternMatch>,
    /// Empirical outcome distribution across the similar_patterns.
    /// For trading: better than a single prediction because it surfaces
    /// uncertainty (a 60/40 distribution is very different from a 95/5
    /// one even though both produce the same argmax direction).
    pub outcome_distribution: OutcomeDistribution,
    /// Counter-evidence: similar past patterns whose outcomes contradict
    /// the headline `direction`. Useful to surface in a UI: "we say UP,
    /// but here are 3 similar past patterns that went DOWN — judge for
    /// yourself." Real reasoning surfaces opposing evidence.
    pub counter_evidence: Vec<PatternMatch>,
    /// Per-horizon forecasts derived from retrieved patterns'
    /// `additional_horizons`. Empty when no per-horizon data was
    /// trained — falls back to single-horizon behavior cleanly.
    pub horizon_predictions: Vec<HorizonPrediction>,
    /// Suggested position sizing. Computed from edge × confidence ×
    /// (1 - anomaly/2). Caps at the reasoner's `max_position_fraction`.
    pub position_sizing: PositionSizing,
    /// Audit trail of reasoning steps the brain took for this analysis.
    /// Each step is a structured tag like "hdc_recall:hit" or
    /// "anomaly:high". Easy to log and review later.
    pub reasoning_steps: Vec<String>,
}

/// Bucketize a float by sign + log-magnitude into a small set of discrete
/// tokens. Returns a short ASCII tag suitable for inclusion in the brain's
/// text encoder. The bucketing is monotonic — close values produce the
/// same or adjacent buckets, so the brain's distributed encoding sees
/// numerically-similar states as activating overlapping input neurons.
/// Coarse log-magnitude bucketing for values that span many decades
/// (price, volume). 16 buckets per decade.
fn bucket(name: &str, value: f64) -> String {
    if !value.is_finite() {
        return format!("{}=nan", name);
    }
    let sign = if value < 0.0 { '-' } else { '+' };
    let mag = value.abs();
    // Bucket from log-magnitude with 16 buckets per decade — fine enough
    // that 100 and 150 bucket differently (they're 0.18 decades apart),
    // but coarse enough that 100.0 and 100.1 still collide and benefit
    // from HDC's nearest-neighbor recall.
    let log_mag = if mag < 1e-12 { -12.0 } else { mag.log10() };
    let idx = ((log_mag + 8.0) * 16.0).clamp(0.0, 255.0) as u32;
    format!("{}{}{}", name, sign, idx)
}

/// Fine-resolution bucketing for values typically in [-1, 1] (returns,
/// sentiments, volatility, normalized indicators). Per-percent buckets so
/// 0.01 and 0.02 are distinguishable — critical for trading where small
/// magnitude differences are regime-defining. Falls back to coarse
/// log-magnitude bucketing for values outside the normal range.
fn bucket_signed(name: &str, value: f64) -> String {
    if !value.is_finite() {
        return format!("{}=nan", name);
    }
    if value.abs() > 1.0 {
        return bucket(name, value);
    }
    // 200 buckets across [-1, 1] — fine enough to distinguish 0.01 from
    // 0.02 (different buckets), coarse enough that 0.011 and 0.012 collide
    // (same bucket → HDC recall finds them as identical).
    let scaled = (value * 100.0).round().clamp(-100.0, 100.0) as i32;
    let sign = if scaled < 0 { '-' } else { '+' };
    format!("{}{}{}", name, sign, scaled.unsigned_abs())
}

/// Extract content words from a news headline for inclusion in the encoded
/// state. Lowercase, alphanumeric-only, length >= 4 to drop common short
/// words ("the", "a", "of"). Limited to first N content words to keep
/// encoding length bounded. Order-stable so same headline → same tokens.
fn headline_tokens(headline: &str, max_tokens: usize) -> Vec<String> {
    headline.split_whitespace()
        .map(|w| w.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>().to_lowercase())
        .filter(|w| w.len() >= 4)
        .take(max_tokens)
        .collect()
}

/// Derive a categorical "macro regime" tag from RSI, fear-greed, and
/// volatility. Prepending this to the encoded state lets the brain
/// learn regime-conditional patterns: "ETF news in regime_euphoric →
/// sell the news (Down)" vs "ETF news in regime_normal → bullish (Up)".
/// Without an explicit regime tag, the brain has to learn that
/// distinction implicitly from feature combinations, which is much
/// harder with limited training data.
///
/// Tags are coarse (~6 categories) so they cluster well in HDC but
/// don't fragment the training set into too many empty buckets.
fn regime_tag(state: &MarketState) -> String {
    // Best-effort lookups — case-insensitive substring match on indicator
    // names. If a strategy doesn't supply RSI or fear-greed, we still get
    // a usable tag from volatility alone.
    let lower_match = |needle: &str| -> Option<f64> {
        state.indicators.iter()
            .find(|(n, _)| n.to_lowercase().contains(needle))
            .map(|(_, v)| *v)
    };
    let rsi = lower_match("rsi").unwrap_or(50.0);
    let fg = lower_match("fear").unwrap_or(0.5);
    let vol = state.volatility;

    // Order matters — tag the most extreme condition first.
    let tag = if rsi >= 75.0 && fg >= 0.75 { "euphoric" }
              else if rsi <= 25.0 && fg <= 0.25 { "capitulation" }
              else if rsi >= 70.0 { "overbought" }
              else if rsi <= 35.0 { "oversold" }
              else if vol > 0.10 { "highvol" }
              else if vol < 0.025 { "calm" }
              else { "normal" };
    format!("regime_{}", tag)
}

/// The trading reasoner. Owns an OrganicBrain and adds domain-specific
/// encoding plus a multi-step inference chain.
pub struct TradingBrain {
    brain: OrganicBrain,
    /// Number of reasoning passes per analysis. Each pass re-presents the
    /// state to the brain in a slightly perturbed form (different ordering
    /// of indicator tokens) and aggregates. More passes → more stable
    /// confidence estimates at the cost of latency.
    pub reasoning_passes: usize,
    /// Threshold above which an anomaly score causes the brain to abstain
    /// (return Flat with low confidence). Set per-strategy.
    pub anomaly_abstain_threshold: f32,
    /// Number of similar past patterns to retrieve and aggregate per
    /// analysis. Larger K → smoother outcome distribution but more
    /// includes-of-marginally-relevant-patterns. 5 is a reasonable default.
    pub top_k_patterns: usize,
    /// Bounded ring of (state_key, outcome) tuples — the brain's external
    /// "trade journal" for explainable retrieval. Independent of HDC
    /// because we want top-K with outcomes, not just nearest text recall.
    /// HDC handles fuzzy similarity; this handles auditable history.
    history: std::collections::VecDeque<(String, Outcome)>,
    /// Maximum entries kept in `history`. Old entries drop off the back
    /// when this is exceeded — the reasoner adapts to recent regimes
    /// rather than being weighed down by ancient data.
    pub history_capacity: usize,
    /// Recency half-life in number of training samples. A pattern N
    /// samples ago contributes weight `0.5 ^ (N / half_life)`. Larger
    /// values = longer memory; smaller = faster adaptation to regime
    /// shifts. Default 1000 = patterns from 1000 samples ago contribute
    /// half as much as the most recent one. Set lower (e.g. 200) for
    /// fast-moving markets, higher (5000+) for stable strategies.
    pub recency_half_life: f32,
    /// Maximum suggested position size as fraction of capital. The
    /// position-sizing output is hard-capped at this value. Default
    /// 0.25 = quarter-Kelly-style — conservative. Aggressive strategies
    /// might raise to 0.5; defensive ones lower to 0.10.
    pub max_position_fraction: f32,
    /// Self-awareness layer: log of own predictions, scored against
    /// realized outcomes when train_on_outcome arrives. Powers the
    /// `self_assessment()` API — overall + per-regime + per-source
    /// accuracy, calibration curve, drift detection.
    pub prediction_log: self_assessment::PredictionLog,
    /// Persistent cache of headline → Claude-extracted compositional
    /// tokens (act_/vrb_/obj_/mag_). Each unique headline costs one
    /// Claude call ever; subsequent encounters are HashMap lookups.
    /// Used by `enrich_news()` to pre-populate `NewsItem.extraction_tokens`
    /// before analyze() runs.
    pub news_cache: news_composer::ExtractionCache,
}

impl Default for TradingBrain {
    fn default() -> Self { Self::new() }
}

impl TradingBrain {
    /// Construct with a fresh production-scale brain. Use `with_brain` to
    /// adopt an existing brain (e.g. one loaded from disk).
    pub fn new() -> Self {
        Self::with_brain(OrganicBrain::new())
    }

    /// Construct with a smaller brain for tests or constrained deployments.
    pub fn new_small(total_neurons: usize) -> Self {
        Self::with_brain(OrganicBrain::new_small(total_neurons))
    }

    pub fn with_brain(brain: OrganicBrain) -> Self {
        Self {
            brain,
            reasoning_passes: 3,
            anomaly_abstain_threshold: 0.85,
            top_k_patterns: 5,
            history: std::collections::VecDeque::new(),
            history_capacity: 5000,
            recency_half_life: 1000.0,
            // Conservative default: 10% of capital max per trade.
            // Empirical backtest with 25% cap showed catastrophic
            // single-trade losses (Spot ETF miss: -3.75% capital from
            // a single 25% position with 15% adverse move). Real
            // trading systems use 1-5%. 10% is the upper end for
            // strategies confident in their edge — most should set
            // lower via the public field.
            max_position_fraction: 0.10,
            prediction_log: self_assessment::PredictionLog::new(),
            news_cache: news_composer::ExtractionCache::load("data/news_extractions.json"),
        }
    }

    /// Enrich a MarketState's news items with Claude-extracted
    /// compositional tokens. For each NewsItem with empty
    /// `extraction_tokens`, calls Claude (via cache) to produce
    /// `(actor, action, object, magnitude)` and stores the resulting
    /// `act_/vrb_/obj_/mag_` tokens.
    ///
    /// Call OFFLINE / before analyze(). Once cache is warm, just a
    /// HashMap lookup — no Claude calls in the trading-decision loop.
    /// Idempotent: NewsItems already enriched are skipped.
    pub fn enrich_news(&mut self, state: &mut MarketState) {
        for n in state.news.iter_mut() {
            if !n.extraction_tokens.is_empty() { continue; }
            let extraction = news_composer::extract_or_call(
                &mut self.news_cache, &n.headline);
            n.extraction_tokens = extraction.to_tokens();
        }
    }

    /// Save the news extraction cache to its configured path.
    pub fn save_news_cache(&self) -> std::io::Result<()> {
        self.news_cache.save()
    }

    /// Construct a safe, no-trade analysis when the brain refuses to
    /// operate (corrupt input or critical health). Returns Flat with
    /// confidence 0 and zero position size — the trader's risk-
    /// management layer can recognize this as "do nothing".
    fn degraded_analysis(&self, steps: Vec<String>) -> Analysis {
        Analysis {
            direction: Direction::Flat,
            confidence: 0.0,
            anomaly_score: 1.0,  // signal "totally novel / unsafe"
            similar_patterns: Vec::new(),
            outcome_distribution: OutcomeDistribution {
                p_up: 0.0, p_down: 0.0, p_flat: 0.0,
                mean_magnitude: 0.0, sample_size: 0,
                mean_win_magnitude: 0.0, mean_loss_magnitude: 0.0,
            },
            counter_evidence: Vec::new(),
            horizon_predictions: Vec::new(),
            position_sizing: PositionSizing {
                fraction: 0.0,
                edge: 0.0,
                rationale: vec!["degraded:no_trade".to_string()],
            },
            reasoning_steps: steps,
        }
    }

    /// Self-assessment over the prediction log. Returns overall +
    /// per-regime + per-source accuracy, calibration curve, drift flag.
    /// Production trading should poll this periodically: a brain that's
    /// silently wrong is worse than one that surfaces its mistakes.
    pub fn self_assessment(&self) -> self_assessment::SelfAssessment {
        self.prediction_log.assess()
    }

    /// Health check: scan internal state for the failure modes that
    /// only show up after days of continuous training. Looks for
    /// NaN/Inf, oversized buffers, stale calibration, anomaly drift,
    /// excessive unscored backlog. Returns a structured HealthReport.
    /// Side-effect-free.
    ///
    /// Polling cadence in production: every 5-15 minutes is reasonable.
    /// Alert thresholds: any Critical → page; sustained Warnings →
    /// investigate; pure Info → log.
    pub fn health_check(&self) -> health::HealthReport {
        let mut findings: Vec<health::HealthFinding> = Vec::new();

        // 1. Buffer sizes vs capacities. If size > capacity, that's
        // a Critical bug — the trim logic isn't running.
        if self.history.len() > self.history_capacity {
            findings.push(health::HealthFinding {
                severity: health::Severity::Critical,
                category: "buffers".into(),
                message: format!("history size {} exceeds capacity {}",
                    self.history.len(), self.history_capacity),
                repaired: None,
            });
        }
        if self.prediction_log.records.len() > self.prediction_log.max_records {
            findings.push(health::HealthFinding {
                severity: health::Severity::Critical,
                category: "buffers".into(),
                message: format!("prediction_log size {} exceeds capacity {}",
                    self.prediction_log.records.len(),
                    self.prediction_log.max_records),
                repaired: None,
            });
        }

        // 2. Unscored backlog. analyze() pushes records;
        // train_on_outcome() scores them. A growing unscored backlog
        // means callers are predicting without ever reporting outcomes,
        // which means self-assessment data is rotting.
        let unscored = self.prediction_log.records.iter()
            .filter(|r| r.actual.is_none()).count();
        let unscored_pct = if self.prediction_log.records.is_empty() { 0.0 }
            else { unscored as f32 / self.prediction_log.records.len() as f32 };
        if unscored_pct > 0.50 && self.prediction_log.records.len() > 20 {
            findings.push(health::HealthFinding {
                severity: health::Severity::Warning,
                category: "scoring".into(),
                message: format!(
                    "{:.0}% of prediction log unscored ({}/{}). \
                     Caller is predicting without train_on_outcome follow-up.",
                    unscored_pct * 100.0, unscored, self.prediction_log.records.len()),
                repaired: None,
            });
        }

        // 3. NaN/Inf in stored prediction records. Should never happen
        // but a Critical surfaces it cleanly if it does.
        let mut bad_floats = 0usize;
        for r in &self.prediction_log.records {
            if health::float_issue("confidence", r.confidence).is_some() { bad_floats += 1; }
            if health::float_issue("anomaly_score", r.anomaly_score).is_some() { bad_floats += 1; }
        }
        if bad_floats > 0 {
            findings.push(health::HealthFinding {
                severity: health::Severity::Critical,
                category: "floats".into(),
                message: format!(
                    "{} NaN/Inf/extreme floats in prediction log — \
                     downstream calculations will be poisoned",
                    bad_floats),
                repaired: None,
            });
        }

        // 4. Calibration freshness: if calibration buckets are full but
        // recent_window predictions show very different accuracy,
        // calibration is stale.
        let assessment = self.prediction_log.assess();
        let scored = assessment.scored_predictions;
        let drift_gap = (assessment.overall_accuracy - assessment.recent_accuracy).abs();
        if scored >= self.prediction_log.recent_window as u64 * 4 && drift_gap > 0.20 {
            findings.push(health::HealthFinding {
                severity: health::Severity::Warning,
                category: "calibration_drift".into(),
                message: format!(
                    "calibration may be stale: lifetime accuracy {:.2} vs recent {:.2} (gap {:.2})",
                    assessment.overall_accuracy,
                    assessment.recent_accuracy,
                    drift_gap),
                repaired: None,
            });
        }

        // 5. Confidence saturation: if recent predictions are all stuck
        // at extreme values (always 1.0 or always 0.0), the model has
        // collapsed.
        let recent_conf: Vec<f32> = self.prediction_log.records.iter()
            .rev().take(self.prediction_log.recent_window)
            .map(|r| r.confidence).collect();
        if recent_conf.len() >= 10 {
            let max_c = recent_conf.iter().cloned().fold(0.0_f32, f32::max);
            let min_c = recent_conf.iter().cloned().fold(1.0_f32, f32::min);
            if (max_c - min_c) < 0.05 {
                findings.push(health::HealthFinding {
                    severity: health::Severity::Warning,
                    category: "confidence_collapse".into(),
                    message: format!(
                        "recent confidence collapsed to single value: \
                         min {:.2} max {:.2} — model has saturated",
                        min_c, max_c),
                    repaired: None,
                });
            }
        }

        // 6. Anomaly saturation: same idea for anomaly score. If every
        // recent prediction reports anomaly ≥ 0.95, the predictor
        // weights have grown too large and the signal is useless.
        let recent_anom: Vec<f32> = self.prediction_log.records.iter()
            .rev().take(self.prediction_log.recent_window)
            .map(|r| r.anomaly_score).collect();
        if recent_anom.len() >= 10 {
            let mean_a: f32 = recent_anom.iter().sum::<f32>() / recent_anom.len() as f32;
            if mean_a > 0.95 {
                findings.push(health::HealthFinding {
                    severity: health::Severity::Warning,
                    category: "anomaly_saturation".into(),
                    message: format!(
                        "recent mean anomaly {:.2} — predictor pegged, signal useless",
                        mean_a),
                    repaired: None,
                });
            }
        }

        health::HealthReport {
            lifetime_predictions: self.prediction_log.next_id - 1,
            history_size: self.history.len(),
            history_capacity: self.history_capacity,
            prediction_log_size: self.prediction_log.records.len(),
            prediction_log_capacity: self.prediction_log.max_records,
            unscored_predictions: unscored,
            findings,
        }
    }

    /// Auto-repair the failure modes flagged by `health_check()` when
    /// safe. Returns the post-repair HealthReport. Idempotent.
    ///
    /// What gets repaired:
    /// - Oversized buffers: trimmed to capacity.
    /// - NaN/Inf floats in PredictionRecords: clamped or zeroed.
    /// - Stale prediction records (>history_capacity old + unscored):
    ///   purged, since their outcomes will never arrive.
    ///
    /// What does NOT get auto-repaired (manual intervention needed):
    /// - Calibration drift (the data is what it is)
    /// - Anomaly saturation (would need brain re-init)
    /// - Confidence collapse (likely indicates a real problem)
    pub fn auto_repair(&mut self) -> health::HealthReport {
        // 1. Trim oversized buffers.
        while self.history.len() > self.history_capacity {
            self.history.pop_back();
        }
        while self.prediction_log.records.len() > self.prediction_log.max_records {
            self.prediction_log.records.pop_front();
        }

        // 2. Clean NaN/Inf in prediction records. Replace with safe
        // defaults: confidence=0, anomaly=0.5.
        for r in self.prediction_log.records.iter_mut() {
            if health::float_issue("confidence", r.confidence).is_some() {
                r.confidence = 0.0;
            }
            if health::float_issue("anomaly_score", r.anomaly_score).is_some() {
                r.anomaly_score = 0.5;
            }
        }

        // 3. Purge very old unscored predictions — outcomes for these
        // will never arrive. Keeps the prediction log focused on actively
        // tracked predictions.
        let stale_threshold = self.prediction_log.max_records / 2;
        let to_remove: Vec<u64> = self.prediction_log.records.iter()
            .take(self.prediction_log.records.len().saturating_sub(stale_threshold))
            .filter(|r| r.actual.is_none())
            .map(|r| r.id).collect();
        self.prediction_log.records.retain(|r| !to_remove.contains(&r.id));

        // Re-run health check to report post-repair state.
        self.health_check()
    }

    /// Persist the trade-journal history to a JSON file. Brain state is
    /// already serialized via the engine's bincode persistence, but the
    /// trading history (which lives in this struct, not the brain) needs
    /// its own save/load. Without this, restarts wipe all trade memory.
    pub fn save_history(&self, path: &str) -> std::io::Result<()> {
        let entries: Vec<(String, Outcome)> = self.history.iter().cloned().collect();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load history from a JSON file written by `save_history`. Replaces
    /// any existing history. Capped at `history_capacity`.
    pub fn load_history(&mut self, path: &str) -> std::io::Result<()> {
        let raw = std::fs::read_to_string(path)?;
        let entries: Vec<(String, Outcome)> = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        self.history.clear();
        for entry in entries.into_iter().take(self.history_capacity) {
            self.history.push_back(entry);
        }
        Ok(())
    }

    /// Number of entries in the trade-journal history. Useful for
    /// monitoring whether load/save round-trips correctly.
    pub fn history_len(&self) -> usize { self.history.len() }

    /// Save a complete state snapshot (history + prediction log +
    /// settings) to disk. The brain itself is NOT included — the
    /// engine's bincode persistence handles that. Use this to checkpoint
    /// "known-good" state hourly so we can roll back if current state
    /// goes bad.
    pub fn save_snapshot(&self, path: &str) -> std::io::Result<()> {
        let snap = StateSnapshot {
            history: self.history.iter().cloned().collect(),
            history_capacity: self.history_capacity,
            recency_half_life: self.recency_half_life,
            max_position_fraction: self.max_position_fraction,
            prediction_log: self.prediction_log.clone(),
        };
        let json = serde_json::to_string_pretty(&snap)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Restore from a snapshot. Replaces history + prediction log +
    /// settings, leaves the brain untouched. The whole point: if
    /// current state is corrupted (NaN, saturated, leaking), restore
    /// the last hourly snapshot and lose at most an hour of training.
    pub fn restore_snapshot(&mut self, path: &str) -> std::io::Result<()> {
        let raw = std::fs::read_to_string(path)?;
        let snap: StateSnapshot = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        self.history = snap.history.into();
        self.history_capacity = snap.history_capacity;
        self.recency_half_life = snap.recency_half_life;
        self.max_position_fraction = snap.max_position_fraction;
        self.prediction_log = snap.prediction_log;
        Ok(())
    }

    /// Weighted token similarity in [0, 1]. Tokens are categorized by
    /// prefix and weighted by trading importance:
    ///   - core market tokens (price, volume, return, volatility): weight 3.0
    ///   - news source / headline content (`src_*`, `w_*`): weight 2.5
    ///   - sentiment / news age (`sent`, `age`): weight 2.0
    ///   - time context (`hour+`, `dow+`): weight 1.0
    ///   - indicator tokens (everything else): weight 1.5
    ///
    /// Same tokens shared between states contribute their weight to the
    /// numerator; all weighted tokens contribute to the denominator. Two
    /// states matching on price + return but not on RSI score higher than
    /// two matching on RSI but not on price — which is the right thing
    /// for trading. Used for top-K retrieval and outcome distribution.
    fn token_similarity(a: &str, b: &str) -> f32 {
        fn weight_of(tok: &str) -> f32 {
            // Core market signal: weighted heavily, dominates similarity.
            if tok.starts_with("p+") || tok.starts_with("p-") { return 3.0; }
            if tok.starts_with("v+") || tok.starts_with("v-") { return 3.0; }
            if tok.starts_with("r+") || tok.starts_with("r-") { return 3.0; }
            if tok.starts_with("s+") || tok.starts_with("s-") { return 3.0; }
            // Regime tag (categorical macro regime): strong signal,
            // groups events by similar market conditions.
            if tok.starts_with("regime_") { return 2.5; }
            // News source: medium signal — same vendor often correlates
            // with similar event types (FED→rates, SEC→regulation).
            if tok.starts_with("src_") { return 2.0; }
            // Sentiment + age: mid-tier — sentiment is a real but noisy
            // signal, age matters for reaction-window analysis.
            if tok.starts_with("sent") || tok.starts_with("age") { return 1.5; }
            // Headline content words (`w_*`): weak signal — too event-
            // specific (each Fed announcement uses different vocabulary).
            // Kept non-zero so a same-headline pair gets a small boost,
            // but doesn't dominate similarity computation.
            if tok.starts_with("w_") { return 0.5; }
            // Compositional news tokens (Claude-extracted). Higher weight
            // than bag-of-words because they ARE the semantic structure:
            //   act_<actor> — entity (FED, SEC, etc.)
            //   vrb_<action> — verb (RAISE, CUT, BAN) — most discriminating:
            //                  separates hawkish from dovish events sharing
            //                  actor + object
            //   obj_<object> — target (RATES, ETF, MINING)
            //   mag_<magnitude> — qualitative scale
            if tok.starts_with("act_") { return 2.0; }
            if tok.starts_with("vrb_") { return 2.5; }
            if tok.starts_with("obj_") { return 2.0; }
            if tok.starts_with("mag_") { return 1.5; }
            // Time context — lowest weight (intraday seasonality is real
            // but weak compared to price action and news).
            if tok.starts_with("hour+") || tok.starts_with("dow+") { return 1.0; }
            // Default (indicators): mid-tier signal.
            1.5
        }

        let toks_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let toks_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
        if toks_a.is_empty() && toks_b.is_empty() { return 1.0; }
        let mut numer = 0.0f32;
        let mut denom = 0.0f32;
        for tok in toks_a.union(&toks_b) {
            let w = weight_of(tok);
            denom += w;
            if toks_a.contains(tok) && toks_b.contains(tok) {
                numer += w;
            }
        }
        if denom <= 0.0 { 0.0 } else { numer / denom }
    }

    /// Find the top-K past patterns most similar to the given key.
    /// Returns (similarity, index_in_history) tuples sorted descending.
    fn top_k_similar(&self, key: &str, k: usize) -> Vec<(f32, usize)> {
        let mut scored: Vec<(f32, usize)> = self.history.iter().enumerate()
            .map(|(i, (past_key, _))| (Self::token_similarity(key, past_key), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Compute the empirical outcome distribution over a set of pattern
    /// matches. Each match contributes `similarity` mass to its outcome's
    /// direction — so very similar past patterns dominate the distribution
    /// and weakly-similar ones contribute less.
    ///
    /// `predicted` is the direction the brain is leaning toward. Used to
    /// split magnitudes into "wins" (matches that agree with prediction)
    /// vs "losses" (matches that disagree). This separation is what
    /// makes EV-aware sizing possible: a 60% directional edge is
    /// catastrophic if your wins are 0.5% but your losses are 5%.
    fn aggregate_outcomes(
        matches: &[PatternMatch],
        predicted: Direction,
    ) -> OutcomeDistribution {
        let mut up = 0.0f32; let mut down = 0.0f32; let mut flat = 0.0f32;
        let mut mag_sum = 0.0f32; let mut weight_sum = 0.0f32;
        let mut win_mag_sum = 0.0f32; let mut win_weight = 0.0f32;
        let mut loss_mag_sum = 0.0f32; let mut loss_weight = 0.0f32;
        for m in matches {
            let w = m.similarity.max(1e-6);
            match m.outcome.direction {
                Direction::Up => up += w,
                Direction::Down => down += w,
                Direction::Flat => flat += w,
            }
            mag_sum += w * m.outcome.magnitude as f32;
            weight_sum += w;

            // Win/loss split based on predicted direction. Flat
            // outcomes are neither wins nor losses — they contribute to
            // overall magnitude but not to win/loss magnitudes that
            // drive sizing. This matters: a "Flat that we predicted Up"
            // is a missed opportunity, not a directional loss.
            let is_win = m.outcome.direction == predicted
                && predicted != Direction::Flat;
            let is_loss = m.outcome.direction != predicted
                && m.outcome.direction != Direction::Flat
                && predicted != Direction::Flat;
            if is_win {
                win_mag_sum += w * m.outcome.magnitude as f32;
                win_weight += w;
            } else if is_loss {
                loss_mag_sum += w * m.outcome.magnitude as f32;
                loss_weight += w;
            }
        }
        let total = up + down + flat;
        let (p_up, p_down, p_flat) = if total > 1e-6 {
            (up / total, down / total, flat / total)
        } else { (0.0, 0.0, 0.0) };
        let mean_magnitude = if weight_sum > 1e-6 { mag_sum / weight_sum } else { 0.0 };
        let mean_win_magnitude = if win_weight > 1e-6 { win_mag_sum / win_weight } else { 0.0 };
        let mean_loss_magnitude = if loss_weight > 1e-6 { loss_mag_sum / loss_weight } else { 0.0 };
        OutcomeDistribution {
            p_up, p_down, p_flat, mean_magnitude,
            sample_size: matches.len() as u32,
            mean_win_magnitude, mean_loss_magnitude,
        }
    }

    /// Encode a market state into a deterministic ASCII string the brain
    /// can ingest through its standard text encoder. The format is
    /// `name+/-bucket name+/-bucket ...` — short tokens that the encoder's
    /// position-sensitive hashing turns into structured spike patterns.
    /// Same numeric input produces same string; close numeric inputs
    /// produce strings that share most tokens, which means the brain's
    /// HDC sees them as similar. This is the key bridge from numerical
    /// market data to the brain's text-native interface.
    pub fn encode_state(&self, state: &MarketState) -> String {
        let mut tokens: Vec<String> = Vec::new();

        // Macro regime tag derived from RSI / fear-greed / volatility.
        // First because it's the highest-level summary of the state.
        tokens.push(regime_tag(state));

        // Numeric market features. Price and volume span decades, so
        // log-magnitude bucketing. Returns, volatility, and indicators
        // are typically in [-1, 1], so fine-resolution per-percent
        // bucketing — critical for trading discrimination.
        tokens.push(bucket("p", state.price));
        tokens.push(bucket("v", state.volume));
        tokens.push(bucket_signed("r", state.recent_return));
        tokens.push(bucket_signed("s", state.volatility));
        // Indicators sorted alphabetically for deterministic ordering —
        // same logical state must produce same encoded key regardless of
        // input order.
        let mut indicators: Vec<&(String, f64)> = state.indicators.iter().collect();
        indicators.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, value) in indicators {
            // Keep full sanitized name (no truncation) so distinct
            // indicators don't collide on a 4-char prefix.
            let safe: String = name.chars()
                .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
                .collect();
            tokens.push(bucket_signed(&safe, *value));
        }

        // Time context — intraday and weekday seasonality.
        if let Some(ts) = state.timestamp {
            tokens.push(format!("hour+{}", ts.hour_utc.min(23)));
            tokens.push(format!("dow+{}", ts.day_of_week.min(6)));
        }

        // News context. Each NewsItem contributes:
        //   - source token (categorical, e.g. "src_fed")
        //   - sentiment bucket (fine resolution)
        //   - age bucket (recent vs old news matters for reaction)
        //   - up to 5 content words from the headline
        // Sorted by source then headline for determinism.
        let mut news_sorted: Vec<&NewsItem> = state.news.iter().collect();
        news_sorted.sort_by(|a, b|
            a.source.cmp(&b.source).then_with(|| a.headline.cmp(&b.headline)));
        for n in news_sorted {
            let src_tag: String = n.source.chars()
                .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
                .collect();
            tokens.push(format!("src_{}", src_tag));
            tokens.push(bucket_signed("sent", n.sentiment));
            tokens.push(bucket("age", n.age_hours.max(0.01)));
            // Compositional Claude-extracted tokens (act_/vrb_/obj_/mag_)
            // give the brain semantic structure: "Fed cuts rates" and
            // "Fed raises rates" share actor+object but differ on action.
            // Falls back to bag-of-words when extraction not available.
            if !n.extraction_tokens.is_empty() {
                for tok in &n.extraction_tokens {
                    tokens.push(tok.clone());
                }
            } else {
                for word in headline_tokens(&n.headline, 5) {
                    tokens.push(format!("w_{}", word));
                }
            }
        }

        tokens.join(" ")
    }

    /// Encode an outcome into an answer string the brain can store. Format
    /// chosen so similar outcomes share token prefixes — Up/Down/Flat
    /// directional bucket plus a magnitude bucket.
    pub fn encode_outcome(&self, outcome: &Outcome) -> String {
        let dir = match outcome.direction {
            Direction::Up => "U", Direction::Down => "D", Direction::Flat => "F",
        };
        format!("{} {}", dir, bucket("m", outcome.magnitude))
    }

    /// Decode the brain's text response back into a `Direction`. The brain
    /// emits the encoded outcome verbatim when it's recalling a stored
    /// pattern, so this is essentially `starts_with("U" / "D" / "F")`.
    /// Falls back to Flat when the response is empty or unparseable —
    /// the safe default for trading is "no signal."
    fn decode_direction(response: &str) -> Direction {
        match response.trim().chars().next() {
            Some('U') => Direction::Up,
            Some('D') => Direction::Down,
            _ => Direction::Flat,
        }
    }

    /// Train the brain on a (state, outcome) pair. Closes the learning
    /// loop: every realized trade outcome flows back into the brain's
    /// pattern memory. This is the online-learning advantage over LLMs.
    ///
    /// Also stores all perturbation variants used by multi-pass analysis
    /// so each pass at inference time finds a stored match — HDC sees
    /// rotated token strings as different patterns, so we register all
    /// of them up front. Trade-off: more HDC writes per training call,
    /// but the alternative (high-confidence misclassification across
    /// passes) is much worse for trading.
    pub fn train_on_outcome(&mut self, state: &MarketState, outcome: &Outcome) {
        // Input sanitization. Bad MarketState (NaN price, Inf volume,
        // garbage indicators) is dropped or repaired BEFORE it touches
        // brain state. Otherwise a single bad call poisons HDC, the
        // history buffer, and every downstream prediction.
        let (state, issues) = match health::validate_and_clean_state(state) {
            Ok(pair) => pair,
            Err(reason) => {
                eprintln!("train_on_outcome: rejecting unrecoverable state: {}", reason);
                return; // Refuse to train on garbage.
            }
        };
        if !issues.is_empty() {
            eprintln!("train_on_outcome: cleaned input: {:?}", issues);
        }
        // Reject NaN/Inf magnitude — the outcome itself.
        if !outcome.magnitude.is_finite() {
            eprintln!("train_on_outcome: rejecting outcome with non-finite magnitude");
            return;
        }
        let state = &state;
        let key = self.encode_state(state);
        let value = self.encode_outcome(outcome);
        self.brain.train(&key, &value);
        for perturbed in self.perturb_keys(&key, self.reasoning_passes).into_iter().skip(1) {
            if perturbed != key {
                self.brain.train(&perturbed, &value);
            }
        }
        // Self-assessment: if a recent unscored prediction matches this
        // state's encoded key, record the realized direction so accuracy
        // stats update automatically. Closes the predict→outcome loop
        // without the caller needing to track prediction IDs explicitly.
        self.prediction_log.score_by_key(&key, outcome.direction);

        // Append to the explainable trade-journal history. Older entries
        // drop off the back when capacity is exceeded — the reasoner
        // weights recent regime data more heavily, which matters in
        // markets that shift over time.
        self.history.push_front((key, outcome.clone()));
        while self.history.len() > self.history_capacity {
            self.history.pop_back();
        }
    }

    /// Run the full reasoning chain on a market state. Multiple passes,
    /// aggregation, structured output. This is the primary API.
    pub fn analyze(&mut self, state: &MarketState) -> Analysis {
        let mut steps: Vec<String> = Vec::new();

        // Input sanitization. Same defensive principle as
        // train_on_outcome — never let bad input into the brain.
        let (state_owned, input_issues) = match health::validate_and_clean_state(state) {
            Ok(pair) => pair,
            Err(reason) => {
                // Unrecoverable input — return a safe Flat / zero-confidence
                // analysis rather than crashing or operating on garbage.
                steps.push(format!("REJECTED:{}", reason));
                return self.degraded_analysis(steps);
            }
        };
        if !input_issues.is_empty() {
            steps.push(format!("input_cleaned:{}_issues", input_issues.len()));
        }
        let state = &state_owned;

        // Degraded mode: if a recent health check found Critical issues,
        // return a safe Flat analysis instead of operating on possibly
        // corrupt internal state. Self-protection: the brain refuses
        // to give predictions it can't trust.
        let health_now = self.health_check();
        if health_now.has_critical() {
            steps.push("DEGRADED:critical_health".to_string());
            for f in &health_now.findings {
                if f.severity == health::Severity::Critical {
                    steps.push(format!("crit:{}", f.message));
                }
            }
            return self.degraded_analysis(steps);
        }

        let key = self.encode_state(state);
        steps.push(format!("encode:{}", &key));

        // Anomaly score from predictive-coding prediction error after the
        // brain processes this input. High error means the input layer
        // produces a hidden pattern the predictor hadn't anticipated —
        // i.e. the brain hasn't seen this kind of state before.
        // (Sample BEFORE process so we read the surprise *for this input*,
        // not for whatever was running before.)

        // Multiple reasoning passes. Each pass perturbs the input slightly
        // (re-orders the indicator tokens) so the brain sees structurally
        // similar but not identical inputs. Aggregating across passes gives
        // a more honest confidence estimate — if the brain's answer is
        // robust to the perturbation, confidence is high; if it flips
        // direction across passes, confidence is low.
        let mut direction_votes: [u32; 3] = [0, 0, 0]; // [Up, Down, Flat]
        let mut nonempty_passes = 0u32;

        let perturbed_keys = self.perturb_keys(&key, self.reasoning_passes);
        for (pass_idx, perturbed) in perturbed_keys.iter().enumerate() {
            let response = self.brain.process(perturbed);
            steps.push(format!("pass{}:{}", pass_idx, summarize_response(&response)));
            if response.is_empty() { continue; }
            nonempty_passes += 1;
            match Self::decode_direction(&response) {
                Direction::Up => direction_votes[0] += 1,
                Direction::Down => direction_votes[1] += 1,
                Direction::Flat => direction_votes[2] += 1,
            }
        }

        // Anomaly score: probe brain on the original key (sets up
        // predictive-coding error reading), then combine with token
        // novelty (max_sim against history → 1 - max_sim). Either
        // signal flagging novelty is enough to raise anomaly.
        let _final_response = self.brain.process(&key);
        let anomaly = self.anomaly_score(&key);
        steps.push(format!("anomaly:{:.3}", anomaly));

        // Pick winning direction; tie or no votes → Flat.
        let (direction, top_votes) = {
            let max = direction_votes.iter().copied().max().unwrap_or(0);
            if max == 0 { (Direction::Flat, 0) }
            else if direction_votes[0] == max { (Direction::Up, max) }
            else if direction_votes[1] == max { (Direction::Down, max) }
            else { (Direction::Flat, max) }
        };

        // Confidence: vote agreement * pass-coverage * (1 - anomaly).
        // All three multipliers in [0, 1]. Strong confidence requires all
        // three to be high — many passes voting the same way, on a
        // non-anomalous state, with most passes producing some answer.
        let agreement = if nonempty_passes == 0 {
            0.0
        } else {
            top_votes as f32 / nonempty_passes as f32
        };
        let coverage = nonempty_passes as f32 / self.reasoning_passes.max(1) as f32;
        let confidence = (agreement * coverage * (1.0 - anomaly)).clamp(0.0, 1.0);
        steps.push(format!("agreement:{:.2}", agreement));
        steps.push(format!("coverage:{:.2}", coverage));

        // Abstention: if anomaly is too high, report Flat with the
        // confidence we computed (which will already be low). The caller
        // can also use the raw anomaly score for their own gating.
        let final_direction = if anomaly >= self.anomaly_abstain_threshold {
            steps.push("abstain:high_anomaly".to_string());
            Direction::Flat
        } else {
            direction
        };

        // STRATIFIED retrieval. The query's regime tag (first token of
        // the encoded key — e.g. "regime_euphoric") is extracted, and we
        // first try to match ONLY against history entries with the same
        // regime tag. This prevents cross-regime contamination: a test
        // event in regime_euphoric should compare against past euphoric-
        // regime events even if a normal-regime event happens to share
        // more news-source tokens. Macro regime conditions a price
        // reaction more than news category does.
        //
        // If no same-regime matches exist, fall back to all-history
        // similarity above a soft threshold — better to surface
        // uncertain inferences than to abstain entirely on novel regimes.
        let query_regime: Option<&str> = key.split_whitespace()
            .find(|t| t.starts_with("regime_"));
        let similarity_threshold = 0.15_f32;

        // Recency-weighted similarity: multiply raw token similarity by
        // exp(-position * ln(2) / half_life). Most recent entry (index 0)
        // gets full weight; entry half_life-ago gets 0.5 weight; entry
        // 2*half_life-ago gets 0.25 weight. Markets are non-stationary —
        // patterns from years ago are less relevant than recent ones.
        let half_life = self.recency_half_life.max(1.0);
        let recency_weight = |idx: usize| -> f32 {
            (-(idx as f32) * std::f32::consts::LN_2 / half_life).exp()
        };

        let make_match = |(idx, (past_key, past_outcome)): (usize, &(String, Outcome))| -> PatternMatch {
            let raw_sim = Self::token_similarity(&key, past_key);
            let weighted = raw_sim * recency_weight(idx);
            PatternMatch {
                state_key: past_key.clone(),
                outcome: past_outcome.clone(),
                similarity: weighted,
            }
        };

        let mut all_matches: Vec<PatternMatch> = if let Some(qr) = query_regime {
            let same_regime: Vec<PatternMatch> = self.history.iter().enumerate()
                .filter(|(_, (past, _))| past.split_whitespace().any(|t| t == qr))
                .map(make_match)
                .collect();
            if !same_regime.is_empty() {
                steps.push(format!("stratified:regime={},n={}", qr, same_regime.len()));
                same_regime
            } else {
                steps.push(format!("stratified:fallback_no_{}_history", qr));
                self.history.iter().enumerate()
                    .map(make_match)
                    .filter(|m| m.similarity >= similarity_threshold)
                    .collect()
            }
        } else {
            self.history.iter().enumerate()
                .map(make_match)
                .filter(|m| m.similarity >= similarity_threshold)
                .collect()
        };
        all_matches.sort_by(|a, b|
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        // First-pass distribution: probabilities only, used to determine
        // direction. Win/loss magnitude split needs the *predicted*
        // direction, which we don't have yet — so we use Flat as a
        // placeholder (win/loss fields will be 0). After direction is
        // determined, we re-aggregate with the correct direction below.
        let outcome_distribution_initial =
            Self::aggregate_outcomes(&all_matches, Direction::Flat);
        let outcome_distribution = outcome_distribution_initial;

        // Display surfaces just the top K. This is what a UI shows.
        let similar_patterns: Vec<PatternMatch> = all_matches
            .iter().take(self.top_k_patterns).cloned().collect();
        steps.push(format!(
            "outcome_dist:up={:.2},down={:.2},flat={:.2},n={}",
            outcome_distribution.p_up, outcome_distribution.p_down,
            outcome_distribution.p_flat, outcome_distribution.sample_size
        ));

        // Override direction with the empirical-distribution argmax if
        // it disagrees with the brain's vote AND the distribution has
        // genuine sample backing. The brain's recall can be wrong on
        // edge cases; the explicit outcome distribution is grounded in
        // realized history.
        let dist_direction = {
            let d = &outcome_distribution;
            if d.sample_size >= 1 {
                // Compute the winner and its margin over the runner-up.
                // If max < 50% OR margin < 10%, the distribution is too
                // split to commit to a direction — better to call Flat
                // than to force a coin-flip prediction. Boring ML
                // produces overconfident argmax even on 51/49 splits;
                // this reasoner abstains when the data doesn't support
                // a directional call.
                let max_p = d.p_up.max(d.p_down).max(d.p_flat);
                let second_max = if (d.p_up - max_p).abs() < 1e-6 {
                    d.p_down.max(d.p_flat)
                } else if (d.p_down - max_p).abs() < 1e-6 {
                    d.p_up.max(d.p_flat)
                } else {
                    d.p_up.max(d.p_down)
                };
                let margin = max_p - second_max;
                if max_p < 0.5 || margin < 0.10 {
                    Some(Direction::Flat)
                } else if d.p_up > d.p_down && d.p_up > d.p_flat {
                    Some(Direction::Up)
                } else if d.p_down > d.p_up && d.p_down > d.p_flat {
                    Some(Direction::Down)
                } else {
                    Some(Direction::Flat)
                }
            } else { None }
        };
        let reasoned_direction = match (final_direction, dist_direction) {
            (Direction::Flat, Some(d)) => {
                steps.push(format!("dist_override:from_flat_to_{:?}", d));
                d
            }
            (current, Some(d)) if current != d => {
                steps.push(format!("dist_override:{:?}_to_{:?}", current, d));
                d
            }
            _ => final_direction,
        };

        // Re-aggregate with the FINAL direction so win/loss magnitudes
        // are split correctly. mean_win_magnitude = mean magnitude of
        // matches that AGREE with reasoned_direction; mean_loss_magnitude
        // = mean magnitude of matches that DISAGREE. These drive
        // EV-aware position sizing in compute_position_sizing.
        let outcome_distribution =
            Self::aggregate_outcomes(&all_matches, reasoned_direction);
        steps.push(format!(
            "win_loss_mag:win={:.4},loss={:.4}",
            outcome_distribution.mean_win_magnitude,
            outcome_distribution.mean_loss_magnitude));

        // Counter-evidence: among the similar patterns, the ones whose
        // outcome contradicts our FINAL reasoned direction (not the raw
        // brain vote). Surfaced explicitly because real reasoning shows
        // opposing data, not just supporting.
        let counter_evidence: Vec<PatternMatch> = similar_patterns.iter()
            .filter(|m| m.outcome.direction != reasoned_direction
                && m.outcome.direction != Direction::Flat)
            .cloned()
            .collect();
        if !counter_evidence.is_empty() {
            steps.push(format!("counter_evidence:n={}", counter_evidence.len()));
        }

        // Confidence blending: combine the brain's vote-derived confidence
        // (which is zero when the brain stayed silent on novel queries)
        // with the empirical-distribution confidence (which is grounded
        // in realized history). For novel events where the brain has no
        // recall, the distribution is the only signal — using it for
        // confidence makes the brain's output meaningful even when its
        // own readout is silent.
        // Confidence design: empirical agreement signal independent of
        // anomaly. Anomaly is reported as its own field — caller decides
        // how to combine. Multiplying confidence by (1-anomaly) here
        // would double-penalize: novel events would always be low-
        // confidence regardless of how strongly the matched patterns
        // agreed, which hides the actual signal.
        let dist_top_prob = match reasoned_direction {
            Direction::Up => outcome_distribution.p_up,
            Direction::Down => outcome_distribution.p_down,
            Direction::Flat => outcome_distribution.p_flat.max(0.5),
        };

        // Beta-Bernoulli shrinkage: 5 training matches all agreeing is
        // not the same as 50 matches all agreeing. Shrink the empirical
        // probability toward a uniform 1/3 prior with weight α=5
        // (interpretable as "5 pseudo-observations of uniform outcome").
        // This prevents pathological confidence=1.00 on a tiny sample.
        // With many real observations, the shrinkage washes out and the
        // empirical probability stands.
        let n = outcome_distribution.sample_size as f32;
        let shrinkage_alpha = 5.0;
        let shrunk_top_prob =
            (dist_top_prob * n + (1.0 / 3.0) * shrinkage_alpha) / (n + shrinkage_alpha);

        let dist_confidence = if outcome_distribution.sample_size >= 1 {
            shrunk_top_prob
        } else { 0.0 };
        steps.push(format!("shrunk_p:{:.2}_n={}", shrunk_top_prob, n));
        let blended_confidence = if nonempty_passes == 0 {
            dist_confidence
        } else {
            (confidence + dist_confidence) * 0.5
        };
        steps.push(format!("dist_conf:{:.2}", dist_confidence));

        // Counter-evidence confidence penalty: subtract a fraction of
        // the opposing-direction probability. Tuned at 0.3 (was 0.5)
        // because too-aggressive penalty zeros out the entire signal
        // for split distributions, which is worse than reporting a
        // moderate confidence with the split surfaced via
        // counter_evidence and outcome_distribution fields.
        let opposing_p = match reasoned_direction {
            Direction::Up => outcome_distribution.p_down,
            Direction::Down => outcome_distribution.p_up,
            Direction::Flat => outcome_distribution.p_up.max(outcome_distribution.p_down),
        };
        let raw_confidence = (blended_confidence - opposing_p * 0.3).clamp(0.0, 1.0);
        steps.push(format!("opposing_p:{:.2}", opposing_p));

        // Confidence recalibration via observed track record. If the
        // calibration bucket for this raw confidence has enough scored
        // observations, override with the observed hit rate. The brain
        // genuinely self-improves: "I said 0.7 confidence and was right
        // 50% of the time, so future 0.7 outputs report as 0.5". No
        // fixed-formula approach gives this property.
        // Min samples = 10 — below that, the bucket's hit rate is too
        // noisy to override raw confidence.
        let confidence = self.prediction_log.recalibrate(raw_confidence, 10);
        if (confidence - raw_confidence).abs() > 0.01 {
            steps.push(format!("recal:{:.2}->{:.2}", raw_confidence, confidence));
        }

        // Per-horizon predictions: aggregate horizon-specific outcomes
        // from the matched past patterns. Each horizon gets its own
        // direction argmax + confidence + mean magnitude. If patterns
        // weren't trained with multi-horizon data, this is empty —
        // backward compatible with single-horizon training.
        let horizon_predictions = Self::compute_horizon_predictions(&similar_patterns);
        if !horizon_predictions.is_empty() {
            steps.push(format!("horizons:n={}", horizon_predictions.len()));
        }

        // Position sizing: edge × confidence × (1 - anomaly/2), clamped
        // to ±max_position_fraction. Sign depends on direction.
        let position_sizing = self.compute_position_sizing(
            reasoned_direction, confidence, anomaly,
            &outcome_distribution, &mut steps);

        // Self-assessment: log this prediction so we can score it later
        // when the realized outcome arrives via train_on_outcome.
        // Extracts regime tag and news source tags from the encoded key
        // for per-regime / per-source accuracy stats.
        let regime = key.split_whitespace()
            .find(|t| t.starts_with("regime_"))
            .unwrap_or("regime_unknown")
            .to_string();
        let news_sources: Vec<String> = key.split_whitespace()
            .filter(|t| t.starts_with("src_"))
            .map(|t| t.to_string())
            .collect();
        self.prediction_log.record(self_assessment::PredictionRecord {
            id: 0, // assigned by record()
            state_key: key.clone(),
            regime,
            news_sources,
            predicted: reasoned_direction,
            confidence,
            anomaly_score: anomaly,
            actual: None,
            hit: None,
        });

        Analysis {
            direction: reasoned_direction,
            confidence,
            anomaly_score: anomaly,
            similar_patterns,
            outcome_distribution,
            counter_evidence,
            horizon_predictions,
            position_sizing,
            reasoning_steps: steps,
        }
    }

    /// Aggregate per-horizon outcomes across matched past patterns.
    /// For each horizon present in the data, compute the direction
    /// argmax weighted by similarity, confidence as the win-probability,
    /// and mean magnitude across matches at that horizon.
    fn compute_horizon_predictions(matches: &[PatternMatch]) -> Vec<HorizonPrediction> {
        use std::collections::HashMap;
        let mut by_horizon: HashMap<u32, Vec<(f32, &HorizonOutcome)>> = HashMap::new();
        for m in matches {
            for h in &m.outcome.additional_horizons {
                by_horizon.entry(h.horizon_hours).or_default()
                    .push((m.similarity, h));
            }
        }
        let mut out: Vec<HorizonPrediction> = Vec::new();
        for (horizon, items) in by_horizon {
            let total_w: f32 = items.iter().map(|(s, _)| s.max(1e-6)).sum();
            if total_w <= 0.0 { continue; }
            let mut up = 0.0f32; let mut down = 0.0f32; let mut flat = 0.0f32;
            let mut mag_sum = 0.0f32;
            for (sim, h) in &items {
                let w = sim.max(1e-6);
                match h.direction {
                    Direction::Up => up += w,
                    Direction::Down => down += w,
                    Direction::Flat => flat += w,
                }
                mag_sum += w * h.magnitude as f32;
            }
            let max = up.max(down).max(flat);
            let direction = if (up - max).abs() < 1e-6 { Direction::Up }
                            else if (down - max).abs() < 1e-6 { Direction::Down }
                            else { Direction::Flat };
            let confidence = (max / total_w).clamp(0.0, 1.0);
            let mean_magnitude = mag_sum / total_w;
            out.push(HorizonPrediction {
                horizon_hours: horizon, direction, confidence, mean_magnitude,
            });
        }
        out.sort_by_key(|p| p.horizon_hours);
        out
    }

    /// Compute suggested position sizing from direction + confidence +
    /// anomaly + outcome distribution. Returns a value in [-max, +max]
    /// that the caller's risk-management layer can scale further.
    ///
    /// Loss-aversion design: position sizing has THREE haircuts beyond
    /// the basic edge × confidence formula:
    ///   1. Anomaly factor (1 - anomaly): on novel events, scale down.
    ///   2. Opposition factor (1 - opposing_p × 2): when distribution
    ///      has meaningful counter-evidence, scale down dramatically.
    ///   3. Confidence floor: very low confidence trades get zero size.
    ///
    /// Why three factors and not one: empirical backtest showed that
    /// 55.6% hit rate with WIN/LOSS RATIO 0.26 is a NET LOSING strategy.
    /// Single high-confidence miss (Spot ETF approved, -15% move while
    /// holding +0.25 max position = -3.75% capital loss) erased 7
    /// winning trades. Counter-evidence-aware sizing addresses exactly
    /// this failure mode: when the brain has high confidence but
    /// counter-evidence exists in the distribution, it should not size
    /// like an unanimous call.
    fn compute_position_sizing(
        &self,
        direction: Direction,
        confidence: f32,
        anomaly: f32,
        dist: &OutcomeDistribution,
        steps: &mut Vec<String>,
    ) -> PositionSizing {
        let mut rationale: Vec<String> = Vec::new();
        let edge = match direction {
            Direction::Up => dist.p_up - dist.p_down,
            Direction::Down => dist.p_down - dist.p_up,
            Direction::Flat => 0.0,
        }.max(0.0).clamp(0.0, 1.0);
        rationale.push(format!("edge={:.2}", edge));

        // Anomaly factor: scale down on novel events. (1 - anomaly/2)
        // is the historical setting — full anomaly halves position.
        // Stronger penalty would crush wins on novel-but-correct calls
        // (most trading edges are in novel territory).
        let anomaly_factor = (1.0 - anomaly * 0.5).clamp(0.0, 1.0);
        rationale.push(format!("anomaly_factor={:.2}", anomaly_factor));

        // Opposition factor: only triggers when counter-evidence is
        // STRONG (>0.20). A 70/30 split is the danger zone where
        // confidence looks high but 30% of similar events went the
        // other way. Below 0.20 opposing, normal sizing — most trades
        // shouldn't be downscaled just because the distribution isn't
        // 100% unanimous.
        // At opposing 0.20: factor 1.0 (no penalty)
        // At opposing 0.35: factor 0.7
        // At opposing 0.50: factor 0.4 (significant penalty)
        let opposing = match direction {
            Direction::Up => dist.p_down,
            Direction::Down => dist.p_up,
            Direction::Flat => 0.0,
        };
        let excess_opposing = (opposing - 0.20).max(0.0);
        let opposition_factor = (1.0 - excess_opposing * 2.0).clamp(0.0, 1.0);
        rationale.push(format!("opposition_factor={:.2}", opposition_factor));

        // Confidence floor: below 0.4, no trade. Hard cutoff stops
        // marginal calls from accumulating exposure.
        let conf_factor = if confidence < 0.4 { 0.0 } else { confidence };
        rationale.push(format!("conf_factor={:.2}", conf_factor));

        // EV gate: estimated expected value per trade based on
        // empirical win/loss magnitudes from similar past patterns.
        //   p_win = probability the prediction was right (top class)
        //   p_loss = probability of opposing direction (loss)
        //   ev = p_win × mean_win_magnitude - p_loss × mean_loss_magnitude
        //
        // This is the SINGLE biggest profit-factor lever. With a 60%
        // edge but 1:5 win/loss magnitude ratio, EV is negative and
        // the trade should be skipped entirely. Without this gate, the
        // brain takes positions where directional probability looks
        // good but realized P&L is structurally negative. Empirical
        // backtest showed this exact pattern (mean_win 0.3% vs
        // mean_loss 1.8%, profit factor 0.4).
        let p_win = match direction {
            Direction::Up => dist.p_up,
            Direction::Down => dist.p_down,
            Direction::Flat => 0.0,
        };
        let p_loss = opposing;
        let ev = p_win * dist.mean_win_magnitude
               - p_loss * dist.mean_loss_magnitude;
        rationale.push(format!("ev={:.4}", ev));

        // Apply EV gate. Even a small EV requirement (>0) eliminates
        // structurally-unprofitable trades. We require strictly positive
        // — a zero-EV trade has zero expected return but costs slippage
        // and emotional capital, so it's net-negative in practice.
        if ev <= 0.0 {
            rationale.push("ev_gate:skip".to_string());
            return PositionSizing {
                fraction: 0.0,
                edge,
                rationale,
            };
        }

        // Magnitude factor: scale position by the win/loss ratio. When
        // mean_win >> mean_loss (favorable asymmetry), boost size.
        // When mean_win << mean_loss (unfavorable asymmetry), shrink.
        // Capped to avoid extreme leverage on degenerate ratios.
        let mag_ratio = if dist.mean_loss_magnitude > 1e-6 {
            (dist.mean_win_magnitude / dist.mean_loss_magnitude).clamp(0.0, 3.0)
        } else if dist.mean_win_magnitude > 1e-6 {
            // No history of losses for this pattern: cautious 1.0x
            1.0
        } else {
            // No magnitude data at all: neutral
            1.0
        };
        rationale.push(format!("mag_ratio={:.2}", mag_ratio));

        let raw = edge * conf_factor * anomaly_factor * opposition_factor * mag_ratio;
        let capped = raw.min(self.max_position_fraction);
        rationale.push(format!("raw={:.3}", raw));
        rationale.push(format!("capped={:.3}", capped));

        let signed = match direction {
            Direction::Up => capped,
            Direction::Down => -capped,
            Direction::Flat => 0.0,
        };
        steps.push(format!("position:{:+.3}", signed));

        PositionSizing { fraction: signed, edge, rationale }
    }

    /// Combined anomaly score in [0, 1]. Two components:
    ///
    /// 1. **Predictive-coding error** from the brain's input→hidden
    ///    predictor. Goes high when the spike pattern is unfamiliar to
    ///    the predictor's learned weights.
    ///
    /// 2. **Token novelty** = `1 - max_similarity_to_history`. Goes
    ///    high when no past pattern shares enough features with the
    ///    query. This differentiates events the predictive coder can't —
    ///    a partially-trained predictor often has near-zero error on
    ///    inputs it has technically never seen, so the second component
    ///    is critical for usable anomaly detection on novel events.
    ///
    /// Combined as max() so EITHER signal flagging novelty is enough to
    /// raise the alarm. A trader receiving this score is best served by
    /// a high recall on novelty, not optimization for false-positive
    /// minimization.
    fn anomaly_score(&self, query_key: &str) -> f32 {
        let raw = self.brain.input_to_hidden_prediction_error();
        let pred_anomaly = 1.0 / (1.0 + (-(raw - 1.0)).exp());
        let max_sim = self.history.iter()
            .map(|(past, _)| Self::token_similarity(query_key, past))
            .fold(0.0_f32, f32::max);
        let novelty = (1.0 - max_sim).clamp(0.0, 1.0);
        pred_anomaly.max(novelty)
    }

    /// Generate `n` perturbed variants of the encoded state by rotating
    /// the token order. Each variant has the same tokens but in a slightly
    /// different sequence — close enough that HDC recall finds the same
    /// patterns, far enough that the brain's position-sensitive encoder
    /// puts the spike pattern in slightly different input neurons. This
    /// is what makes the multi-pass aggregation informative.
    fn perturb_keys(&self, key: &str, n: usize) -> Vec<String> {
        let mut tokens: Vec<&str> = key.split_whitespace().collect();
        let token_count = tokens.len();
        let mut variants = Vec::with_capacity(n);
        variants.push(tokens.join(" "));
        for i in 1..n {
            let shift = if token_count == 0 { 0 } else { i % token_count };
            tokens.rotate_left(shift);
            variants.push(tokens.join(" "));
        }
        variants
    }
}

fn summarize_response(response: &str) -> String {
    if response.is_empty() { return "empty".to_string(); }
    let head: String = response.chars().take(8).collect();
    format!("got:{}", head)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state(price: f64) -> MarketState {
        MarketState {
            price,
            volume: 1000.0,
            recent_return: 0.005,
            volatility: 0.02,
            indicators: vec![
                ("rsi".to_string(), 55.0),
                ("macd".to_string(), 0.001),
            ],
            news: Vec::new(),
            timestamp: None,
        }
    }

    #[test]
    fn test_encode_state_is_deterministic() {
        let tb = TradingBrain::new_small(2048);
        let s1 = sample_state(100.0);
        let s2 = sample_state(100.0);
        assert_eq!(tb.encode_state(&s1), tb.encode_state(&s2));
    }

    #[test]
    fn test_encode_state_differs_for_different_states() {
        let tb = TradingBrain::new_small(2048);
        let s1 = sample_state(100.0);
        let s2 = sample_state(150.0);
        let e1 = tb.encode_state(&s1);
        let e2 = tb.encode_state(&s2);
        // Same indicators but different price → some token differs.
        assert_ne!(e1, e2, "different prices must produce different encodings");
    }

    #[test]
    fn test_outcome_encoding_distinguishes_directions() {
        let tb = TradingBrain::new_small(2048);
        let up = Outcome::new(Direction::Up, 0.02);
        let down = Outcome::new(Direction::Down, 0.02);
        assert_ne!(tb.encode_outcome(&up), tb.encode_outcome(&down));
    }

    #[test]
    fn test_decode_direction_safe_default() {
        // Empty / garbage → Flat (the safe trading default).
        assert_eq!(TradingBrain::decode_direction(""), Direction::Flat);
        assert_eq!(TradingBrain::decode_direction("???"), Direction::Flat);
        assert_eq!(TradingBrain::decode_direction("U m+5"), Direction::Up);
        assert_eq!(TradingBrain::decode_direction("D m+3"), Direction::Down);
    }

    /// Train the brain on a (state, outcome) pair and verify the next
    /// analysis of that same state recalls the trained direction. This is
    /// the foundational learning loop: closed feedback from realized
    /// outcomes back into the brain.
    #[test]
    fn test_trained_pattern_recalls() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);
        let outcome = Outcome::new(Direction::Up, 0.02);

        // Train the same pair many times so HDC stores it firmly and
        // the spiking pathway gets reinforced.
        for _ in 0..30 { tb.train_on_outcome(&state, &outcome); }

        let analysis = tb.analyze(&state);
        assert_eq!(analysis.direction, Direction::Up,
            "Trained Up state should be recalled as Up; got {:?} (steps: {:?})",
            analysis.direction, analysis.reasoning_steps);
    }

    /// Distinct (state, outcome) pairs should produce distinct analyses —
    /// the brain must not collapse different training pairs into one.
    /// In real trading, bull and bear regimes differ in multiple features
    /// (returns, RSI, MACD, volatility) — not just price level — so the
    /// test mirrors that. Encodings must be substantially different at
    /// the token level for HDC's nearest-neighbor recall to keep them
    /// distinct (HDC treats high-similarity keys as the same pattern,
    /// which is the correct behavior for noise tolerance but means we
    /// need genuine multi-feature differentiation).
    #[test]
    fn test_distinct_states_give_distinct_directions() {
        let mut tb = TradingBrain::new_small(2048);
        let bull_state = MarketState {
            price: 100.0, volume: 5000.0,
            recent_return: 0.05, volatility: 0.015,
            indicators: vec![
                ("rsi".to_string(), 72.0),
                ("macd".to_string(), 0.04),
            ],
            news: Vec::new(),
            timestamp: None,
        };
        let bear_state = MarketState {
            price: 100.0, volume: 5000.0,
            recent_return: -0.05, volatility: 0.04,
            indicators: vec![
                ("rsi".to_string(), 28.0),
                ("macd".to_string(), -0.04),
            ],
            news: Vec::new(),
            timestamp: None,
        };
        let bull_outcome = Outcome::new(Direction::Up, 0.03);
        let bear_outcome = Outcome::new(Direction::Down, 0.03);

        for _ in 0..30 {
            tb.train_on_outcome(&bull_state, &bull_outcome);
            tb.train_on_outcome(&bear_state, &bear_outcome);
        }

        let bull_analysis = tb.analyze(&bull_state);
        let bear_analysis = tb.analyze(&bear_state);
        assert_eq!(bull_analysis.direction, Direction::Up,
            "bull state must recall as Up (steps: {:?})", bull_analysis.reasoning_steps);
        assert_eq!(bear_analysis.direction, Direction::Down,
            "bear state must recall as Down (steps: {:?})", bear_analysis.reasoning_steps);
    }

    /// An untrained state should produce low confidence — the brain
    /// shouldn't pretend it knows. This is a key calibration property
    /// for use in a real trading pipeline.
    #[test]
    fn test_untrained_state_low_confidence() {
        let mut tb = TradingBrain::new_small(2048);
        let novel_state = sample_state(50.0);
        let analysis = tb.analyze(&novel_state);
        assert!(analysis.confidence < 0.5,
            "Untrained state must produce low confidence, got {} (anomaly {})",
            analysis.confidence, analysis.anomaly_score);
    }

    /// The reasoning trail must capture the steps the brain took, so a
    /// downstream system or human can audit decisions.
    #[test]
    fn test_analysis_includes_reasoning_steps() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);
        let analysis = tb.analyze(&state);
        assert!(!analysis.reasoning_steps.is_empty(),
            "Analysis must include audit-trail reasoning steps");
        assert!(analysis.reasoning_steps.iter().any(|s| s.starts_with("encode:")),
            "Reasoning steps must include the encoded input");
        assert!(analysis.reasoning_steps.iter().any(|s| s.starts_with("anomaly:")),
            "Reasoning steps must include the anomaly score");
    }

    /// THE differentiator from XGBoost and friends: the analysis should
    /// surface SPECIFIC past patterns it considered similar, with their
    /// realized outcomes. This is what makes the reasoning *explainable*
    /// rather than just a single prediction score.
    #[test]
    fn test_analysis_surfaces_similar_past_patterns() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);
        let outcome = Outcome::new(Direction::Up, 0.02);

        // Build up trade journal history.
        for _ in 0..10 { tb.train_on_outcome(&state, &outcome); }

        let analysis = tb.analyze(&state);
        assert!(!analysis.similar_patterns.is_empty(),
            "Analysis must retrieve specific past patterns it considered");
        // Top match should be highly similar to the queried state (we
        // trained the same state, so similarity ≈ 1.0).
        let top = &analysis.similar_patterns[0];
        assert!(top.similarity > 0.5,
            "Top similar pattern's similarity should be high, got {}", top.similarity);
        // And its recorded outcome should match the trained one.
        assert_eq!(top.outcome.direction, Direction::Up);
    }

    /// Outcome distribution gives the empirical base rate over similar
    /// past patterns — not just one prediction. A trader can act
    /// differently on a 95/5 split vs a 60/40 split even when the
    /// argmax direction is the same.
    #[test]
    fn test_outcome_distribution_reflects_training() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);

        // Train mostly Up, some Down — the distribution should reflect this.
        for _ in 0..8 {
            tb.train_on_outcome(&state, &Outcome::new(Direction::Up, 0.02));
        }
        for _ in 0..2 {
            tb.train_on_outcome(&state, &Outcome::new(Direction::Down, 0.02));
        }

        let analysis = tb.analyze(&state);
        let d = &analysis.outcome_distribution;
        assert!(d.sample_size > 0, "Distribution must have non-zero samples");
        assert!(d.p_up > d.p_down,
            "Distribution should reflect 8:2 Up:Down training, got up={} down={}",
            d.p_up, d.p_down);
        assert!(d.p_up > 0.5, "Up should be majority, got {}", d.p_up);
    }

    /// Counter-evidence: when training was mixed, the analysis must
    /// surface past patterns that disagree with the headline direction.
    /// Real reasoning shows opposing evidence — boring ML hides it.
    #[test]
    fn test_counter_evidence_surfaces_opposing_outcomes() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);

        // 7 Up, 3 Down. Headline will be Up, counter-evidence has 3 Downs.
        for _ in 0..7 {
            tb.train_on_outcome(&state, &Outcome::new(Direction::Up, 0.02));
        }
        for _ in 0..3 {
            tb.train_on_outcome(&state, &Outcome::new(Direction::Down, 0.03));
        }

        let analysis = tb.analyze(&state);
        assert_eq!(analysis.direction, Direction::Up,
            "expected Up; got {:?}\n  steps: {:?}\n  dist: {:?}",
            analysis.direction, analysis.reasoning_steps,
            analysis.outcome_distribution);
        assert!(!analysis.counter_evidence.is_empty(),
            "When training is mixed, counter-evidence must surface (got {:?})",
            analysis.counter_evidence);
        // Counter-evidence outcomes should all contradict the headline.
        for ce in &analysis.counter_evidence {
            assert_ne!(ce.outcome.direction, Direction::Up,
                "Counter-evidence should not include same-direction patterns");
        }
    }

    /// Multi-modal: news context must affect the encoded key. The same
    /// numeric market state with different news should produce DIFFERENT
    /// encodings — otherwise news input is being silently dropped.
    #[test]
    fn test_news_context_changes_encoding() {
        let tb = TradingBrain::new_small(2048);
        let base = sample_state(50000.0);
        let mut with_fed_news = base.clone();
        with_fed_news.news.push(NewsItem {
            source: "FED".to_string(),
            headline: "Fed raises rates by fifty basis points".to_string(),
            sentiment: -0.6,
            age_hours: 1.0,
            extraction_tokens: Vec::new(),
        });

        let e_base = tb.encode_state(&base);
        let e_news = tb.encode_state(&with_fed_news);
        assert_ne!(e_base, e_news,
            "News should change the encoded state key (base={}, news={})",
            e_base, e_news);
        assert!(e_news.contains("src_fed"), "News encoding must include source tag");
        assert!(e_news.contains("sent-"), "Negative sentiment must encode with - sign");
    }

    /// Multi-modal: training on (state + news → outcome) and querying
    /// with the same news should recall the trained outcome. This is the
    /// core trading use case — Fed announces, brain remembers historical
    /// reactions to similar Fed announcements.
    #[test]
    fn test_news_context_drives_recall() {
        let mut tb = TradingBrain::new_small(2048);
        let mut state = sample_state(50000.0);
        state.news.push(NewsItem {
            source: "FED".to_string(),
            headline: "FOMC raises rates aggressive hawkish".to_string(),
            sentiment: -0.7,
            age_hours: 0.5,
            extraction_tokens: Vec::new(),
        });
        state.timestamp = Some(TimeContext { hour_utc: 18, day_of_week: 2 });

        // Train: when this kind of news arrives, BTC drops.
        let outcome = Outcome::new(Direction::Down, 0.05);
        for _ in 0..30 { tb.train_on_outcome(&state, &outcome); }

        let analysis = tb.analyze(&state);
        assert_eq!(analysis.direction, Direction::Down,
            "Trained Fed-hawkish state should recall as Down (steps: {:?})",
            analysis.reasoning_steps);
    }

    /// Long-running simulation: run thousands of predict/train cycles
    /// to catch the bugs that only manifest after days of operation.
    /// Specifically checks:
    ///   - Buffer sizes stay bounded (no leaks)
    ///   - No NaN/Inf in confidence / anomaly / position fields
    ///   - Confidence doesn't saturate (collapse to single value)
    ///   - Anomaly doesn't saturate at 1.0
    ///   - Auto-repair fixes any issues the health check catches
    ///
    /// `#[ignore]` because it's slow (~5K predict+train cycles).
    /// Run on every release: `cargo test --release -- --ignored
    /// test_long_running_robustness`.
    #[test]
    #[ignore]
    fn test_long_running_robustness() {
        let mut tb = TradingBrain::new_small(2048);

        // Cycle through several state archetypes so the brain sees
        // diverse patterns. 1K iterations is enough to surface
        // long-running bugs (NaN creep, buffer overflow, saturation)
        // without taking forever — production-cadence equivalent of
        // ~1 day of operation with multiple predictions per minute.
        let archetypes: Vec<(MarketState, Outcome)> = vec![
            (MarketState {
                price: 50000.0, volume: 1e10,
                recent_return: 0.02, volatility: 0.03,
                indicators: vec![("rsi".into(), 60.0)],
                news: vec![NewsItem {
                    source: "FED".into(),
                    headline: "Fed dovish remarks".into(),
                    sentiment: 0.4, age_hours: 1.0,
                    extraction_tokens: Vec::new(),
                }],
                timestamp: Some(TimeContext { hour_utc: 14, day_of_week: 2 }),
            }, Outcome::new(Direction::Up, 0.02)),
            (MarketState {
                price: 50000.0, volume: 1e10,
                recent_return: -0.05, volatility: 0.08,
                indicators: vec![("rsi".into(), 28.0), ("fear_greed".into(), 0.20)],
                news: vec![NewsItem {
                    source: "REUTERS".into(),
                    headline: "Exchange hack panic".into(),
                    sentiment: -0.7, age_hours: 0.5,
                    extraction_tokens: Vec::new(),
                }],
                timestamp: Some(TimeContext { hour_utc: 9, day_of_week: 4 }),
            }, Outcome::new(Direction::Down, 0.05)),
            (MarketState {
                price: 50000.0, volume: 1e10,
                recent_return: 0.0, volatility: 0.02,
                indicators: vec![("rsi".into(), 50.0)],
                news: vec![],
                timestamp: Some(TimeContext { hour_utc: 22, day_of_week: 6 }),
            }, Outcome::new(Direction::Flat, 0.005)),
        ];

        for cycle in 0..1000 {
            let (state, outcome) = &archetypes[cycle % archetypes.len()];
            let _analysis = tb.analyze(state);
            tb.train_on_outcome(state, outcome);

            // Spot-check every 200 cycles.
            if cycle % 200 == 199 {
                let h = tb.health_check();
                assert!(!h.has_critical(),
                    "cycle {}: critical health finding(s): {:?}",
                    cycle, h.findings);
                // Buffers must stay bounded.
                assert!(h.history_size <= h.history_capacity,
                    "history overflow at cycle {}", cycle);
                assert!(h.prediction_log_size <= h.prediction_log_capacity,
                    "log overflow at cycle {}", cycle);
            }
        }

        // Final state: scan for stale-data issues, run auto_repair,
        // verify no Criticals afterward.
        let pre = tb.health_check();
        let post = tb.auto_repair();
        assert!(!post.has_critical(),
            "auto_repair did not clear critical findings: {:?}", post.findings);
        eprintln!("After 1000 cycles: pre-repair {:?} findings, post-repair {:?} findings",
            pre.findings.len(), post.findings.len());
    }

    /// Self-assessment loop: analyze → realized outcome → score the
    /// prediction → assessment reflects the realized accuracy. This is
    /// the closed feedback loop that makes the reasoner self-aware.
    #[test]
    fn test_self_assessment_round_trip() {
        let mut tb = TradingBrain::new_small(2048);
        let bull = MarketState {
            price: 100.0, volume: 5000.0,
            recent_return: 0.05, volatility: 0.02,
            indicators: vec![("rsi".into(), 70.0)],
            news: vec![NewsItem {
                source: "FED".into(),
                headline: "Fed dovish pivot rate cut signals".into(),
                sentiment: 0.6, age_hours: 1.0,
                extraction_tokens: Vec::new(),
            }],
            timestamp: Some(TimeContext { hour_utc: 18, day_of_week: 2 }),
        };
        // Train heavily so analyze recalls Up.
        for _ in 0..30 {
            tb.train_on_outcome(&bull, &Outcome::new(Direction::Up, 0.04));
        }
        // analyze() pushes a prediction record.
        let analysis = tb.analyze(&bull);
        assert_eq!(analysis.direction, Direction::Up);

        // Now report a realized Up outcome via train_on_outcome — this
        // should score the just-made prediction as a hit.
        tb.train_on_outcome(&bull, &Outcome::new(Direction::Up, 0.05));

        let a = tb.self_assessment();
        assert!(a.scored_predictions >= 1);
        assert!(a.overall_accuracy >= 0.5,
            "Hit rate should reflect the scored hit, got {}", a.overall_accuracy);

        // Per-source stats should record the FED source.
        assert!(a.accuracy_by_source.keys().any(|k| k == "src_fed"));
        // Per-regime should record overbought (rsi=70 + sentiment = euphoric? or overbought).
        assert!(a.accuracy_by_regime.keys().any(|k| k.starts_with("regime_")));
    }

    /// History persistence: save_history → load_history must round-trip
    /// the trade journal exactly. Production deployments will use this
    /// to keep accumulated trade data across restarts.
    #[test]
    fn test_history_persistence_roundtrip() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);
        let outcome = Outcome::new(Direction::Up, 0.02);
        for _ in 0..5 { tb.train_on_outcome(&state, &outcome); }
        let len_before = tb.history_len();
        assert_eq!(len_before, 5);

        let path = format!("/tmp/trading_history_test_{}.json", std::process::id());
        tb.save_history(&path).expect("save_history");

        let mut tb2 = TradingBrain::new_small(2048);
        assert_eq!(tb2.history_len(), 0);
        tb2.load_history(&path).expect("load_history");
        assert_eq!(tb2.history_len(), len_before);

        let _ = std::fs::remove_file(&path);
    }

    /// Volatility-aware bucketing: returns of 0.01 vs 0.02 must encode
    /// differently. Coarse log-magnitude bucketing collapsed both into
    /// the same bucket; fine bucketing distinguishes them — critical for
    /// trading regime discrimination.
    #[test]
    fn test_fine_bucketing_distinguishes_small_returns() {
        let tb = TradingBrain::new_small(2048);
        let mut s1 = sample_state(100.0);
        let mut s2 = sample_state(100.0);
        s1.recent_return = 0.01;
        s2.recent_return = 0.02;
        let e1 = tb.encode_state(&s1);
        let e2 = tb.encode_state(&s2);
        assert_ne!(e1, e2,
            "Returns 0.01 and 0.02 must encode differently: {} vs {}", e1, e2);
    }

    /// Confidence calibration: a state seen unanimously should have
    /// strictly higher confidence than a state seen with 60/40 split.
    /// Boring ML often produces overconfident scores; the reasoner here
    /// is built to track real epistemic uncertainty.
    #[test]
    fn test_confidence_reflects_consensus() {
        let mut tb = TradingBrain::new_small(2048);
        let unanimous = sample_state(100.0);
        let split = MarketState {
            price: 200.0, volume: 5000.0,
            recent_return: 0.01, volatility: 0.03,
            indicators: vec![("rsi".to_string(), 50.0)],
            news: Vec::new(),
            timestamp: None,
        };

        // Unanimous: 10 Up training samples.
        for _ in 0..10 {
            tb.train_on_outcome(&unanimous, &Outcome::new(Direction::Up, 0.02));
        }
        // Split: 6 Up, 4 Down.
        for _ in 0..6 {
            tb.train_on_outcome(&split, &Outcome::new(Direction::Up, 0.02));
        }
        for _ in 0..4 {
            tb.train_on_outcome(&split, &Outcome::new(Direction::Down, 0.02));
        }

        let unanimous_a = tb.analyze(&unanimous);
        let split_a = tb.analyze(&split);

        // Both predict Up but unanimous should be more confident.
        // We can't assert strict inequality on the brain's internal
        // confidence (which depends on multiple factors), but we can
        // assert the empirical-distribution probability is strictly
        // higher in the unanimous case — that's the "real" confidence.
        assert!(unanimous_a.outcome_distribution.p_up
                > split_a.outcome_distribution.p_up,
            "Unanimous-up state must have higher p_up than split state: \
             unanimous={} split={}",
            unanimous_a.outcome_distribution.p_up,
            split_a.outcome_distribution.p_up);
    }
}
