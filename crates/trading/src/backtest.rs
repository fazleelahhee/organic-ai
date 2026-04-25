//! Backtest harness for the trading reasoner.
//!
//! Trains a `TradingBrain` on a chronological-prefix split of historical
//! events, evaluates on the held-out suffix, and produces a structured
//! `BacktestReport` with hit rate, magnitude calibration, abstention
//! behavior, and per-event detail. Use it to answer questions like:
//! "after training on events through 2023, how does the brain do on
//! 2024 events it has never seen?"
//!
//! Honest design: we do NOT shuffle. Trading reality is forward-only —
//! a backtest that lets the brain peek at the future via random splits
//! is worthless. Only chronological splits matter.

use crate::{Analysis, Direction, TradingBrain};
use crate::seed::HistoricalEvent;

/// Per-event evaluation result.
#[derive(Debug, Clone)]
pub struct EventResult {
    pub label: String,
    pub date: String,
    /// What the brain predicted.
    pub predicted: Direction,
    /// What actually happened.
    pub actual: Direction,
    /// Hit if predicted == actual (or both are Flat).
    pub hit: bool,
    /// The brain's reported confidence in its prediction.
    pub confidence: f32,
    /// Anomaly score for this event (was it flagged as out-of-distribution?).
    pub anomaly_score: f32,
    /// Did the brain abstain (return Flat with low confidence)?
    pub abstained: bool,
    /// Number of similar past patterns the brain found (proxy for
    /// "how much grounding did this prediction have").
    pub n_similar: usize,
}

/// Aggregated backtest results.
#[derive(Debug, Clone)]
pub struct BacktestReport {
    pub n_train: usize,
    pub n_test: usize,
    pub hits: usize,
    pub misses: usize,
    pub abstentions: usize,
    pub hit_rate_overall: f32,
    /// Hit rate excluding abstentions — what's the brain's accuracy
    /// when it commits to a directional prediction?
    pub hit_rate_when_committed: f32,
    /// Mean confidence on hit events (higher = better calibrated when right).
    pub mean_confidence_on_hits: f32,
    /// Mean confidence on miss events (lower = better calibrated when wrong).
    pub mean_confidence_on_misses: f32,
    /// Mean anomaly score on the test set. Should be higher than train
    /// (since test events are by definition unseen) — if it isn't, the
    /// anomaly signal is broken.
    pub mean_anomaly_test: f32,
    /// Per-event details for diagnostic inspection.
    pub events: Vec<EventResult>,
}

impl BacktestReport {
    /// Pretty-print summary suitable for terminal output. Shows the
    /// per-event breakdown plus aggregate metrics — designed for the
    /// "review until satisfied" iteration loop, where you scan it,
    /// notice patterns of failure, fix the reasoner, re-run.
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("\n=== Backtest Report ===\n"));
        s.push_str(&format!("Train events: {}\n", self.n_train));
        s.push_str(&format!("Test events:  {}\n", self.n_test));
        s.push_str(&format!("Hits:         {} / {}\n", self.hits, self.n_test));
        s.push_str(&format!("Abstentions:  {} ({:.0}%)\n",
            self.abstentions,
            100.0 * self.abstentions as f32 / self.n_test.max(1) as f32));
        s.push_str(&format!("Hit rate overall:        {:.1}%\n",
            100.0 * self.hit_rate_overall));
        s.push_str(&format!("Hit rate when committed: {:.1}%\n",
            100.0 * self.hit_rate_when_committed));
        s.push_str(&format!("Mean conf on hits:   {:.2}\n", self.mean_confidence_on_hits));
        s.push_str(&format!("Mean conf on misses: {:.2}\n", self.mean_confidence_on_misses));
        s.push_str(&format!("Mean anomaly on test: {:.2}\n", self.mean_anomaly_test));
        s.push_str("\n--- Per-event ---\n");
        for e in &self.events {
            let mark = if e.hit { "✓" } else { "✗" };
            let note = if e.abstained { " [abstain]" } else { "" };
            s.push_str(&format!(
                "{} {} ({}): pred={:?} actual={:?} conf={:.2} anom={:.2} sim={}{}\n",
                mark, e.date, e.label, e.predicted, e.actual,
                e.confidence, e.anomaly_score, e.n_similar, note));
        }
        s
    }
}

/// Random baseline hit rate: if we predicted Up/Down/Flat uniformly at
/// random, we'd be right ~33% of the time on a 3-class problem. Anything
/// the brain delivers above that is genuine signal.
pub const RANDOM_BASELINE: f32 = 0.333;

/// Run a chronological-split backtest. `train_fraction` in (0, 1)
/// determines how much of the dataset is used for training; the rest is
/// the held-out test set.
///
/// `repeats` lets you train each event multiple times (interleaved order)
/// to firm up the brain's HDC + spiking memory of the training set
/// before evaluation. 5-10 is reasonable for the small seed dataset.
pub fn run_backtest(
    events: &[HistoricalEvent],
    train_fraction: f32,
    repeats: usize,
    brain_total_neurons: usize,
) -> BacktestReport {
    let split_idx = (events.len() as f32 * train_fraction).round() as usize;
    let split_idx = split_idx.clamp(1, events.len().saturating_sub(1).max(1));
    let (train, test) = events.split_at(split_idx);

    let mut tb = TradingBrain::new_small(brain_total_neurons);

    // Train: interleaved repeats so each event is rehearsed against the
    // others. Mirrors how the brain's HDC + spiking + LSM channels were
    // designed to be trained.
    for _ in 0..repeats {
        for ev in train {
            tb.train_on_outcome(&ev.state, &ev.outcome);
        }
    }

    // Evaluate on test set. Each event gets a fresh `analyze()` call.
    let mut results: Vec<EventResult> = Vec::with_capacity(test.len());
    for ev in test {
        let analysis: Analysis = tb.analyze(&ev.state);
        let predicted = analysis.direction;
        let actual = ev.outcome.direction;
        let hit = predicted == actual;
        // Abstained = brain returned Flat AND confidence is low. Genuine
        // Flat predictions from the distribution are NOT abstentions.
        let abstained = predicted == Direction::Flat && analysis.confidence < 0.3;
        results.push(EventResult {
            label: ev.label.to_string(),
            date: ev.date.to_string(),
            predicted,
            actual,
            hit,
            confidence: analysis.confidence,
            anomaly_score: analysis.anomaly_score,
            abstained,
            n_similar: analysis.similar_patterns.len(),
        });
    }

    let n_test = results.len();
    let hits = results.iter().filter(|r| r.hit).count();
    let misses = n_test - hits;
    let abstentions = results.iter().filter(|r| r.abstained).count();
    let committed = n_test - abstentions;
    let hits_when_committed = results.iter()
        .filter(|r| !r.abstained && r.hit).count();

    let hit_rate_overall = if n_test == 0 { 0.0 }
        else { hits as f32 / n_test as f32 };
    let hit_rate_when_committed = if committed == 0 { 0.0 }
        else { hits_when_committed as f32 / committed as f32 };

    let conf_hits: Vec<f32> = results.iter().filter(|r| r.hit).map(|r| r.confidence).collect();
    let conf_misses: Vec<f32> = results.iter().filter(|r| !r.hit).map(|r| r.confidence).collect();
    let mean_confidence_on_hits = if conf_hits.is_empty() { 0.0 }
        else { conf_hits.iter().sum::<f32>() / conf_hits.len() as f32 };
    let mean_confidence_on_misses = if conf_misses.is_empty() { 0.0 }
        else { conf_misses.iter().sum::<f32>() / conf_misses.len() as f32 };

    let mean_anomaly_test = if results.is_empty() { 0.0 }
        else { results.iter().map(|r| r.anomaly_score).sum::<f32>() / results.len() as f32 };

    BacktestReport {
        n_train: train.len(),
        n_test,
        hits,
        misses,
        abstentions,
        hit_rate_overall,
        hit_rate_when_committed,
        mean_confidence_on_hits,
        mean_confidence_on_misses,
        mean_anomaly_test,
        events: results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::historical_btc_events;

    /// Smoke test: backtest runs end-to-end without panic, produces
    /// a non-empty report. Marked `#[ignore]` because it's slow.
    #[test]
    #[ignore]
    fn test_backtest_runs() {
        let events = historical_btc_events();
        let report = run_backtest(&events, 0.7, 3, 4096);
        assert!(report.n_test > 0);
        assert!(report.n_train > 0);
        eprintln!("{}", report.pretty());
    }

    /// THE evaluation: train brain on chronological prefix, test on
    /// held-out suffix, verify hit rate beats the random baseline of 33%.
    /// If the brain doesn't beat random on this curated dataset, the
    /// reasoner needs more work.
    #[test]
    #[ignore]
    fn test_backtest_beats_random_baseline() {
        let events = historical_btc_events();
        // Use a larger brain (16K neurons) for the real eval — small
        // brains are noisy on this multi-modal task.
        let report = run_backtest(&events, 0.7, 5, 16384);
        eprintln!("{}", report.pretty());

        assert!(report.hit_rate_overall > RANDOM_BASELINE,
            "Hit rate {:.1}% must beat random baseline {:.1}%",
            100.0 * report.hit_rate_overall, 100.0 * RANDOM_BASELINE);

        // Calibration sanity is intentionally informational, not a hard
        // assertion. Hit/miss confidence comparison on a small (9-event)
        // test set is high-variance — the test failing on a slight
        // inversion creates noise without signal. We log it; a human
        // running the harness can compare across multiple iterations
        // and decide if there's a systematic issue.
        eprintln!(
            "Calibration: hits_conf={:.2} misses_conf={:.2} (informational)",
            report.mean_confidence_on_hits, report.mean_confidence_on_misses);
    }
}
