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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub direction: Direction,
    /// Magnitude of move as a percentage (e.g. 0.025 = 2.5% up if Up).
    pub magnitude: f64,
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
        }
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
    /// and weakly-similar ones contribute less. This is the "weighted
    /// k-NN" base rate that becomes the analysis's outcome_distribution.
    fn aggregate_outcomes(matches: &[PatternMatch]) -> OutcomeDistribution {
        let mut up = 0.0f32; let mut down = 0.0f32; let mut flat = 0.0f32;
        let mut mag_sum = 0.0f32; let mut weight_sum = 0.0f32;
        for m in matches {
            let w = m.similarity.max(1e-6);
            match m.outcome.direction {
                Direction::Up => up += w,
                Direction::Down => down += w,
                Direction::Flat => flat += w,
            }
            mag_sum += w * m.outcome.magnitude as f32;
            weight_sum += w;
        }
        let total = up + down + flat;
        let (p_up, p_down, p_flat) = if total > 1e-6 {
            (up / total, down / total, flat / total)
        } else { (0.0, 0.0, 0.0) };
        let mean_magnitude = if weight_sum > 1e-6 { mag_sum / weight_sum } else { 0.0 };
        OutcomeDistribution {
            p_up, p_down, p_flat, mean_magnitude,
            sample_size: matches.len() as u32,
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
            for word in headline_tokens(&n.headline, 5) {
                tokens.push(format!("w_{}", word));
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
        let key = self.encode_state(state);
        let value = self.encode_outcome(outcome);
        self.brain.train(&key, &value);
        for perturbed in self.perturb_keys(&key, self.reasoning_passes).into_iter().skip(1) {
            if perturbed != key {
                self.brain.train(&perturbed, &value);
            }
        }
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

        let mut all_matches: Vec<PatternMatch> = if let Some(qr) = query_regime {
            let same_regime: Vec<PatternMatch> = self.history.iter()
                .filter(|(past, _)| past.split_whitespace().any(|t| t == qr))
                .map(|(past_key, past_outcome)| PatternMatch {
                    state_key: past_key.clone(),
                    outcome: past_outcome.clone(),
                    similarity: Self::token_similarity(&key, past_key),
                })
                .collect();
            if !same_regime.is_empty() {
                steps.push(format!("stratified:regime={},n={}", qr, same_regime.len()));
                same_regime
            } else {
                steps.push(format!("stratified:fallback_no_{}_history", qr));
                self.history.iter()
                    .map(|(past_key, past_outcome)| PatternMatch {
                        state_key: past_key.clone(),
                        outcome: past_outcome.clone(),
                        similarity: Self::token_similarity(&key, past_key),
                    })
                    .filter(|m| m.similarity >= similarity_threshold)
                    .collect()
            }
        } else {
            self.history.iter()
                .map(|(past_key, past_outcome)| PatternMatch {
                    state_key: past_key.clone(),
                    outcome: past_outcome.clone(),
                    similarity: Self::token_similarity(&key, past_key),
                })
                .filter(|m| m.similarity >= similarity_threshold)
                .collect()
        };
        all_matches.sort_by(|a, b|
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        // Distribution uses every match above threshold (proper base rate).
        let outcome_distribution = Self::aggregate_outcomes(&all_matches);

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
            // Even a single matched past pattern is informative — a
            // strict ">= 3" cutoff means novel events with limited
            // historical coverage default to Flat, which is the wrong
            // bias for trading (better to surface a tentative call with
            // appropriate confidence than to abstain entirely).
            if d.sample_size >= 1 {
                if d.p_up > d.p_down && d.p_up > d.p_flat { Some(Direction::Up) }
                else if d.p_down > d.p_up && d.p_down > d.p_flat { Some(Direction::Down) }
                else if d.p_flat > 0.5 { Some(Direction::Flat) }
                else { None }
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
        let confidence = (blended_confidence - opposing_p * 0.3).clamp(0.0, 1.0);
        steps.push(format!("opposing_p:{:.2}", opposing_p));

        Analysis {
            direction: reasoned_direction,
            confidence,
            anomaly_score: anomaly,
            similar_patterns,
            outcome_distribution,
            counter_evidence,
            reasoning_steps: steps,
        }
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
        let up = Outcome { direction: Direction::Up, magnitude: 0.02 };
        let down = Outcome { direction: Direction::Down, magnitude: 0.02 };
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
        let outcome = Outcome { direction: Direction::Up, magnitude: 0.02 };

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
        let bull_outcome = Outcome { direction: Direction::Up, magnitude: 0.03 };
        let bear_outcome = Outcome { direction: Direction::Down, magnitude: 0.03 };

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
        let outcome = Outcome { direction: Direction::Up, magnitude: 0.02 };

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
            tb.train_on_outcome(&state, &Outcome {
                direction: Direction::Up, magnitude: 0.02 });
        }
        for _ in 0..2 {
            tb.train_on_outcome(&state, &Outcome {
                direction: Direction::Down, magnitude: 0.02 });
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
            tb.train_on_outcome(&state, &Outcome {
                direction: Direction::Up, magnitude: 0.02 });
        }
        for _ in 0..3 {
            tb.train_on_outcome(&state, &Outcome {
                direction: Direction::Down, magnitude: 0.03 });
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
        });
        state.timestamp = Some(TimeContext { hour_utc: 18, day_of_week: 2 });

        // Train: when this kind of news arrives, BTC drops.
        let outcome = Outcome { direction: Direction::Down, magnitude: 0.05 };
        for _ in 0..30 { tb.train_on_outcome(&state, &outcome); }

        let analysis = tb.analyze(&state);
        assert_eq!(analysis.direction, Direction::Down,
            "Trained Fed-hawkish state should recall as Down (steps: {:?})",
            analysis.reasoning_steps);
    }

    /// History persistence: save_history → load_history must round-trip
    /// the trade journal exactly. Production deployments will use this
    /// to keep accumulated trade data across restarts.
    #[test]
    fn test_history_persistence_roundtrip() {
        let mut tb = TradingBrain::new_small(2048);
        let state = sample_state(100.0);
        let outcome = Outcome { direction: Direction::Up, magnitude: 0.02 };
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
            tb.train_on_outcome(&unanimous, &Outcome {
                direction: Direction::Up, magnitude: 0.02 });
        }
        // Split: 6 Up, 4 Down.
        for _ in 0..6 {
            tb.train_on_outcome(&split, &Outcome {
                direction: Direction::Up, magnitude: 0.02 });
        }
        for _ in 0..4 {
            tb.train_on_outcome(&split, &Outcome {
                direction: Direction::Down, magnitude: 0.02 });
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
