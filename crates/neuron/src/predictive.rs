/// Predictive coding layer.
///
/// Maintains a learned linear model that predicts the *next* compressed
/// neural state from the *current* one.  The prediction error drives
/// curiosity and modulates learning — exactly the free-energy principle.

use serde::{Deserialize, Serialize};

const PRED_DIM: usize = 256;

/// A single prediction layer (linear model + error tracking).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionLayer {
    /// Weight matrix flattened: PRED_DIM x PRED_DIM.
    pub(crate) weights: Vec<f32>,
    /// Most recent scalar prediction error.
    pub prediction_error: f32,
    /// Learning rate.
    pub(crate) lr: f32,
}

impl PredictionLayer {
    pub fn new() -> Self {
        // Small random init via deterministic LCG
        let mut state: u64 = 9876543210;
        let mut weights = Vec::with_capacity(PRED_DIM * PRED_DIM);
        for _ in 0..PRED_DIM * PRED_DIM {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((state >> 33) as f32) / (u32::MAX as f32);
            weights.push((t - 0.5) * 0.02);
        }
        Self {
            weights,
            prediction_error: 0.0,
            lr: 0.001,
        }
    }

    /// Stride-sample neuron firing rates into a fixed-size compressed vector.
    pub fn compress(&self, rates: &[f32], layer_offset: usize) -> Vec<f32> {
        let n = rates.len().saturating_sub(layer_offset);
        if n == 0 {
            return vec![0.0; PRED_DIM];
        }
        let stride = (n / PRED_DIM).max(1);
        let mut out = Vec::with_capacity(PRED_DIM);
        for i in 0..PRED_DIM {
            let idx = layer_offset + i * stride;
            out.push(rates.get(idx).copied().unwrap_or(0.0));
        }
        out
    }

    /// Linear prediction from current compressed state.
    pub fn predict(&self, current: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(PRED_DIM);
        for o in 0..PRED_DIM {
            let base = o * PRED_DIM;
            let mut sum = 0.0f32;
            for i in 0..current.len().min(PRED_DIM) {
                sum += self.weights[base + i] * current[i];
            }
            out.push(sum);
        }
        out
    }

    /// Predict next, compare with actual, delta-rule update.
    /// Returns the scalar mean-squared prediction error.
    pub fn update(&mut self, current: &[f32], actual_next: &[f32]) -> f32 {
        let predicted = self.predict(current);
        let mut total_err = 0.0f32;
        for o in 0..PRED_DIM.min(actual_next.len()) {
            let err = actual_next[o] - predicted[o];
            total_err += err * err;
            let base = o * PRED_DIM;
            for i in 0..current.len().min(PRED_DIM) {
                self.weights[base + i] += self.lr * err * current[i];
            }
        }
        self.prediction_error = total_err / PRED_DIM as f32;
        self.prediction_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_length() {
        let layer = PredictionLayer::new();
        let rates = vec![0.5f32; 1000];
        let c = layer.compress(&rates, 0);
        assert_eq!(c.len(), PRED_DIM);
    }

    #[test]
    fn test_predict_length() {
        let layer = PredictionLayer::new();
        let input = vec![0.1f32; PRED_DIM];
        let pred = layer.predict(&input);
        assert_eq!(pred.len(), PRED_DIM);
    }

    #[test]
    fn test_error_decreases_with_repeated_input() {
        let mut layer = PredictionLayer::new();
        let current = vec![0.3f32; PRED_DIM];
        let next = vec![0.7f32; PRED_DIM];

        let err_first = layer.update(&current, &next);
        for _ in 0..200 {
            layer.update(&current, &next);
        }
        let err_last = layer.update(&current, &next);

        assert!(
            err_last < err_first,
            "Prediction error should decrease: first={} last={}",
            err_first,
            err_last
        );
    }
}
