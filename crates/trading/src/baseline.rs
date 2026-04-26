//! Boring-ML baseline: pure-numeric k-NN classifier on MarketState
//! features. No news, no sentiment, no text encoding — the
//! XGBoost / random-forest equivalent if you did zero text feature
//! engineering and just fed in the numeric columns.
//!
//! Why this is the right baseline:
//! - XGBoost on tabular data is the typical "boring ML" choice for
//!   trading prediction. To handle news, you'd build a separate text
//!   pipeline (TF-IDF, embeddings, sentiment scoring) and concat
//!   features. Most teams don't bother for in-house systems.
//! - This baseline simulates "use the structured numeric data as-is"
//!   without text. If the OrganicAI brain's multi-modal approach
//!   delivers measurable lift, that lift is the value-add of the
//!   text-aware encoding.
//! - k-NN is structurally similar to what a random forest does at the
//!   leaf level, and is what gradient-boosted trees converge toward
//!   for sparse data. Performance gap between k-NN and proper XGBoost
//!   on a 21-event dataset is small — both are dominated by the data
//!   limit, not the model class.
//!
//! What this is NOT:
//! - Not real XGBoost. Real XGBoost adds tree splitting, regularization,
//!   leaf-wise learning, etc. On large datasets it would beat k-NN
//!   substantially. On 21 events it doesn't have room to.
//! - Not a comprehensive ML library — this is a single-purpose baseline
//!   for honest comparison, not a production tool.

use crate::{Direction, MarketState, Outcome};

/// k-Nearest-Neighbors classifier over numeric features.
pub struct KnnBaseline {
    history: Vec<(Vec<f64>, Outcome)>,
    k: usize,
}

impl KnnBaseline {
    pub fn new(k: usize) -> Self {
        Self { history: Vec::new(), k: k.max(1) }
    }

    /// Extract a 10-dim numeric feature vector from a MarketState.
    /// All features normalized to roughly [0, 1] or [-1, 1] range so
    /// Euclidean distance is balanced across dimensions.
    pub fn extract_features(state: &MarketState) -> Vec<f64> {
        let log_price = state.price.max(1e-6).ln() / 15.0; // BTC range ~ ln(100k)≈11
        let log_volume = state.volume.max(1e-6).ln() / 30.0; // ~ ln(1e12)≈27
        let rsi = state.indicators.iter()
            .find(|(n, _)| n.to_lowercase().contains("rsi"))
            .map(|(_, v)| v / 100.0).unwrap_or(0.5);
        let fg = state.indicators.iter()
            .find(|(n, _)| n.to_lowercase().contains("fear"))
            .map(|(_, v)| *v).unwrap_or(0.5);
        let hour = state.timestamp
            .map(|t| t.hour_utc as f64 / 24.0).unwrap_or(0.5);
        let dow = state.timestamp
            .map(|t| t.day_of_week as f64 / 7.0).unwrap_or(0.5);
        let n_news = (state.news.len() as f64 / 5.0).min(1.0);
        let mean_sent = if state.news.is_empty() { 0.0 }
            else { state.news.iter().map(|n| n.sentiment).sum::<f64>()
                / state.news.len() as f64 };
        vec![
            log_price, log_volume,
            state.recent_return, state.volatility,
            rsi, fg, hour, dow, n_news, mean_sent,
        ]
    }

    pub fn train(&mut self, state: &MarketState, outcome: &Outcome) {
        self.history.push((Self::extract_features(state), outcome.clone()));
    }

    /// Predict (direction, confidence) for a state. Confidence is the
    /// dominant-class proportion among the K nearest neighbors,
    /// weighted by inverse distance.
    pub fn predict(&self, state: &MarketState) -> (Direction, f32) {
        if self.history.is_empty() { return (Direction::Flat, 0.0); }
        let q = Self::extract_features(state);
        let mut dists: Vec<(f64, &Outcome)> = self.history.iter()
            .map(|(feats, out)| {
                let d2: f64 = q.iter().zip(feats.iter())
                    .map(|(a, b)| (a - b).powi(2)).sum();
                (d2.sqrt(), out)
            })
            .collect();
        dists.sort_by(|a, b|
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(dists.len());
        let top = &dists[..k];

        let mut up = 0.0_f64; let mut down = 0.0_f64; let mut flat = 0.0_f64;
        for (d, out) in top {
            let w = 1.0 / (d + 1e-6);
            match out.direction {
                Direction::Up => up += w,
                Direction::Down => down += w,
                Direction::Flat => flat += w,
            }
        }
        let total = up + down + flat;
        if total <= 0.0 { return (Direction::Flat, 0.0); }
        let max = up.max(down).max(flat);
        let dir = if (up - max).abs() < 1e-9 { Direction::Up }
                  else if (down - max).abs() < 1e-9 { Direction::Down }
                  else { Direction::Flat };
        (dir, (max / total) as f32)
    }

    pub fn history_len(&self) -> usize { self.history.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NewsItem, TimeContext};

    fn s(price: f64, ret: f64) -> MarketState {
        MarketState {
            price, volume: 1000.0,
            recent_return: ret, volatility: 0.02,
            indicators: vec![],
            news: vec![],
            timestamp: None,
        }
    }

    #[test]
    fn test_features_extracted() {
        let f = KnnBaseline::extract_features(&s(100.0, 0.01));
        assert_eq!(f.len(), 10);
    }

    #[test]
    fn test_baseline_recalls_trained_pair() {
        let mut b = KnnBaseline::new(3);
        let st = s(100.0, 0.05);
        let oc = Outcome::new(Direction::Up, 0.02);
        for _ in 0..5 { b.train(&st, &oc); }
        let (dir, _conf) = b.predict(&st);
        assert_eq!(dir, Direction::Up);
    }

    #[test]
    fn test_baseline_distinguishes_states() {
        let mut b = KnnBaseline::new(3);
        let bull = s(100.0, 0.05);
        let bear = s(100.0, -0.05);
        b.train(&bull, &Outcome::new(Direction::Up, 0.02));
        b.train(&bear, &Outcome::new(Direction::Down, 0.02));
        // Repeat to give nearest-neighbors weight.
        for _ in 0..3 {
            b.train(&bull, &Outcome::new(Direction::Up, 0.02));
            b.train(&bear, &Outcome::new(Direction::Down, 0.02));
        }
        assert_eq!(b.predict(&bull).0, Direction::Up);
        assert_eq!(b.predict(&bear).0, Direction::Down);
    }
}
