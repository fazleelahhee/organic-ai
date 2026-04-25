/// Attention via gain modulation.
///
/// Projects input patterns to queries and hidden firing rates to keys,
/// computes a dot-product relevance score, and produces per-neuron
/// multiplicative gains in [0.5, 2.0].  This lets the brain selectively
/// amplify or suppress hidden-layer activity — organic attention.

use serde::{Deserialize, Serialize};

const ATTENTION_DIM: usize = 512;

/// Gain-modulation attention module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionModule {
    /// Projects input pattern → query vector (ATTENTION_DIM x input_sample).
    pub(crate) query_weights: Vec<f32>,
    /// Projects hidden rates → key vector (ATTENTION_DIM x hidden_sample).
    pub(crate) key_weights: Vec<f32>,
    /// Current global gain (diagnostic).
    pub gain: f32,
    /// Stride used to sample input / hidden populations.
    pub(crate) sample_stride: usize,
    /// Number of input samples kept.
    input_samples: usize,
    /// Number of hidden samples kept.
    hidden_samples: usize,
}

impl AttentionModule {
    pub fn new(input_pop: usize, hidden_pop: usize, seed: u64) -> Self {
        let sample_stride = 64.max(1);
        let input_samples = (input_pop / sample_stride).max(1).min(ATTENTION_DIM);
        let hidden_samples = (hidden_pop / sample_stride).max(1).min(ATTENTION_DIM);

        let mut state = seed;
        let fast_randf = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((*s >> 33) as f32) / (u32::MAX as f32);
            (t - 0.5) * 0.02
        };

        let mut query_weights = Vec::with_capacity(ATTENTION_DIM * input_samples);
        for _ in 0..ATTENTION_DIM * input_samples {
            query_weights.push(fast_randf(&mut state));
        }

        let mut key_weights = Vec::with_capacity(ATTENTION_DIM * hidden_samples);
        for _ in 0..ATTENTION_DIM * hidden_samples {
            key_weights.push(fast_randf(&mut state));
        }

        Self {
            query_weights,
            key_weights,
            gain: 1.0,
            sample_stride,
            input_samples,
            hidden_samples,
        }
    }

    /// Compute per-neuron multiplicative gains for the hidden population.
    ///
    /// Returns a `Vec<f32>` of length `hidden_pop` with values in [0.5, 2.0].
    pub fn compute_gains(
        &mut self,
        input_pattern: &[f32],
        hidden_rates: &[f32],
        hidden_pop: usize,
    ) -> Vec<f32> {
        // --- query from input ---
        let mut query = vec![0.0f32; ATTENTION_DIM];
        for q in 0..ATTENTION_DIM {
            let base = q * self.input_samples;
            let mut sum = 0.0f32;
            for s in 0..self.input_samples {
                let idx = s * self.sample_stride;
                let v = input_pattern.get(idx).copied().unwrap_or(0.0);
                sum += self.query_weights[base + s] * v;
            }
            query[q] = sum;
        }

        // --- key from hidden ---
        let mut key = vec![0.0f32; ATTENTION_DIM];
        for k in 0..ATTENTION_DIM {
            let base = k * self.hidden_samples;
            let mut sum = 0.0f32;
            for s in 0..self.hidden_samples {
                let idx = s * self.sample_stride;
                let v = hidden_rates.get(idx).copied().unwrap_or(0.0);
                sum += self.key_weights[base + s] * v;
            }
            key[k] = sum;
        }

        // --- dot product → scalar relevance ---
        let dot: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
        // Normalise into a rough [0, 1] range via sigmoid
        let relevance = 1.0 / (1.0 + (-dot).exp());
        self.gain = relevance;

        // Map relevance to per-neuron gains.
        // Neurons whose firing-rate pattern aligns with the key direction
        // get amplified; others get suppressed.
        let mut gains = Vec::with_capacity(hidden_pop);
        for i in 0..hidden_pop {
            // Blend base gain with per-neuron modulation
            let rate = hidden_rates.get(i).copied().unwrap_or(0.0);
            // Neurons that are already active AND relevant get boosted
            let g = 0.5 + 1.5 * relevance * (0.3 + 0.7 * rate);
            gains.push(g.clamp(0.5, 2.0));
        }
        gains
    }

    /// Perturbation-based learning: nudge weights in proportion to prediction error.
    pub fn learn(
        &mut self,
        input: &[f32],
        hidden_rates: &[f32],
        prediction_error: f32,
        _hidden_pop: usize,
    ) {
        let lr = 0.0001 * prediction_error.clamp(0.0, 1.0);
        // Perturb query weights toward higher-energy input directions
        for q in 0..ATTENTION_DIM {
            let base = q * self.input_samples;
            for s in 0..self.input_samples {
                let idx = s * self.sample_stride;
                let v = input.get(idx).copied().unwrap_or(0.0);
                self.query_weights[base + s] += lr * v;
            }
        }
        // Perturb key weights toward active hidden directions
        for k in 0..ATTENTION_DIM {
            let base = k * self.hidden_samples;
            for s in 0..self.hidden_samples {
                let idx = s * self.sample_stride;
                let v = hidden_rates.get(idx).copied().unwrap_or(0.0);
                self.key_weights[base + s] += lr * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gains_in_range() {
        let mut attn = AttentionModule::new(1000, 2000, 77);
        let input = vec![0.5f32; 1000];
        let hidden = vec![0.3f32; 2000];
        let gains = attn.compute_gains(&input, &hidden, 2000);
        assert_eq!(gains.len(), 2000);
        for &g in &gains {
            assert!(g >= 0.5 && g <= 2.0, "gain {} out of [0.5, 2.0]", g);
        }
    }

    #[test]
    fn test_similar_inputs_similar_gains() {
        let mut attn = AttentionModule::new(1000, 2000, 77);
        let hidden = vec![0.3f32; 2000];

        let input_a = vec![0.5f32; 1000];
        let mut input_b = vec![0.5f32; 1000];
        // Slightly perturb a few values
        input_b[0] = 0.51;
        input_b[10] = 0.49;

        let gains_a = attn.compute_gains(&input_a, &hidden, 2000);
        let gains_b = attn.compute_gains(&input_b, &hidden, 2000);

        let diff: f32 = gains_a
            .iter()
            .zip(gains_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / gains_a.len() as f32;
        assert!(
            diff < 0.1,
            "Similar inputs should produce similar gains, got avg diff {}",
            diff
        );
    }
}
