/// Liquid State Machine readout layer.
///
/// Samples a subset of hidden-layer neurons, reads their firing rates,
/// and projects them through a learned linear layer to produce a
/// lower-dimensional state vector.  Training uses the delta rule —
/// no backpropagation through the reservoir.

use serde::{Deserialize, Serialize};

pub const SAMPLE_SIZE: usize = 4096;
/// 1024 = 10 character positions × 95 printable-ASCII chars + 74 unused slack.
/// Used by `OrganicBrain` as a position-conditioned char-classification readout.
pub const OUTPUT_DIM: usize = 1024;
pub const CHARS_PER_POSITION: usize = 95;
pub const MAX_OUTPUT_LEN: usize = 10;

/// Readout layer that taps into hidden-population firing rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LsmReadout {
    /// Indices into the hidden population that we sample.
    pub(crate) sample_indices: Vec<u32>,
    /// Weight matrix flattened: OUTPUT_DIM rows x SAMPLE_SIZE cols.
    pub(crate) weights: Vec<f32>,
    /// Bias per output unit.
    pub(crate) bias: Vec<f32>,
    /// Learning rate for the delta rule.
    pub(crate) lr: f32,
}

impl LsmReadout {
    /// Create a readout layer that samples `SAMPLE_SIZE` neurons from a
    /// hidden population of `hidden_pop_size` neurons.
    pub fn new(hidden_pop_size: usize, seed: u64) -> Self {
        let mut state = seed;
        let fast_rand = |s: &mut u64, max: usize| -> usize {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as usize) % max
        };
        let fast_randf = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((*s >> 33) as f32) / (u32::MAX as f32);
            (t - 0.5) * 0.02 // small random in [-0.01, 0.01]
        };

        let mut sample_indices = Vec::with_capacity(SAMPLE_SIZE);
        for _ in 0..SAMPLE_SIZE {
            sample_indices.push(fast_rand(&mut state, hidden_pop_size) as u32);
        }

        let mut weights = Vec::with_capacity(OUTPUT_DIM * SAMPLE_SIZE);
        for _ in 0..OUTPUT_DIM * SAMPLE_SIZE {
            weights.push(fast_randf(&mut state));
        }

        let bias = vec![0.0f32; OUTPUT_DIM];

        Self { sample_indices, weights, bias, lr: 0.001 }
    }

    /// Read firing rates from the sampled hidden neurons.
    pub fn collect_state(&self, neurons: &[crate::brain::BrainNeuron]) -> Vec<f32> {
        self.sample_indices
            .iter()
            .map(|&idx| {
                neurons
                    .get(idx as usize)
                    .map(|n| n.firing_rate())
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Linear projection + sigmoid → Vec<f32> of len OUTPUT_DIM.
    pub fn forward(&self, state: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(OUTPUT_DIM);
        for o in 0..OUTPUT_DIM {
            let base = o * SAMPLE_SIZE;
            let mut sum = self.bias[o];
            for s in 0..state.len().min(SAMPLE_SIZE) {
                sum += self.weights[base + s] * state[s];
            }
            // sigmoid
            output.push(1.0 / (1.0 + (-sum).exp()));
        }
        output
    }

    /// Delta-rule weight update:  dw = lr * (target - output) * state
    pub fn train(&mut self, state: &[f32], target: &[f32]) {
        let output = self.forward(state);
        for o in 0..OUTPUT_DIM.min(target.len()) {
            let err = target[o] - output[o];
            let base = o * SAMPLE_SIZE;
            for s in 0..state.len().min(SAMPLE_SIZE) {
                self.weights[base + s] += self.lr * err * state[s];
            }
            self.bias[o] += self.lr * err;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_produces_output() {
        let readout = LsmReadout::new(10000, 123);
        let state = vec![0.5f32; SAMPLE_SIZE];
        let out = readout.forward(&state);
        assert_eq!(out.len(), OUTPUT_DIM);
        // All outputs should be in (0, 1) because of sigmoid
        for &v in &out {
            assert!(v > 0.0 && v < 1.0, "output {} not in (0,1)", v);
        }
    }

    #[test]
    fn test_training_reduces_error() {
        let mut readout = LsmReadout::new(10000, 42);
        let state = vec![0.3f32; SAMPLE_SIZE];
        let target = vec![0.8f32; OUTPUT_DIM];

        let out_before = readout.forward(&state);
        let err_before: f32 = out_before
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (t - o).powi(2))
            .sum();

        for _ in 0..50 {
            readout.train(&state, &target);
        }

        let out_after = readout.forward(&state);
        let err_after: f32 = out_after
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (t - o).powi(2))
            .sum();

        assert!(
            err_after < err_before,
            "Error should decrease: before={} after={}",
            err_before,
            err_after
        );
    }
}
