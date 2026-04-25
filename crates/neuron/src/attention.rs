/// Attention via gain modulation.
///
/// Two paths now coexist:
///
/// 1. **Global gain (legacy)**: projects input patterns to queries and
///    hidden firing rates to keys, computes a single dot-product, applies
///    the same scaled gain to every hidden neuron. One-shot, content-light.
///
/// 2. **Per-column content-dependent gain (new, 2026-04-25)**: each
///    cortical column carries a learned attention key (a 256-dim vector).
///    At each tick, the brain compresses the current input frame into the
///    same 256-dim space and computes per-column similarity. High-similarity
///    columns get amplified; low-similarity columns get suppressed.
///
///    Hebbian-style learning: when a column fires above its baseline, its
///    attention key drifts toward the current input. Over experience, each
///    column develops a "preferred input" — the inputs it's been firing
///    for. This is the inductive bias for compositional reasoning: at
///    inference, columns whose preferred inputs match parts of a novel
///    query light up, even if that exact query was never seen before.
///
///    This is the SNN analogue of transformer attention's "which tokens
///    matter for this query" mechanism, restricted to per-column granularity
///    so it's tractable at 80M-neuron scale (8 columns × 256 dims = 2048
///    keys instead of one key per neuron).

use serde::{Deserialize, Serialize};

const ATTENTION_DIM: usize = 512;
/// Compressed input dimension used by per-column attention keys.
pub const COL_KEY_DIM: usize = 256;

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
    /// One learned attention key per cortical column. Each key is a
    /// COL_KEY_DIM-dim vector in the same compressed input space the brain
    /// projects each tick's frame into. At inference, similarity between
    /// the current frame's compression and each column's key produces a
    /// per-column gain. At learning, active columns drift their keys
    /// toward the current input — Hebbian "neurons that fire together
    /// wire together," at the column level.
    #[serde(default)]
    pub(crate) column_keys: Vec<Vec<f32>>,
    /// Stride into the input population for compressing a frame to
    /// COL_KEY_DIM. Set at construction so input_pop / col_input_stride ≈
    /// COL_KEY_DIM. Stored so compression is consistent across calls.
    #[serde(default)]
    pub(crate) col_input_stride: usize,
}

impl AttentionModule {
    pub fn new(input_pop: usize, hidden_pop: usize, seed: u64) -> Self {
        Self::new_with_columns(input_pop, hidden_pop, 1, seed)
    }

    pub fn new_with_columns(
        input_pop: usize,
        hidden_pop: usize,
        num_columns: usize,
        seed: u64,
    ) -> Self {
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

        // Per-column attention keys, small random init.
        let mut column_keys: Vec<Vec<f32>> = Vec::with_capacity(num_columns);
        for _ in 0..num_columns {
            let mut k = Vec::with_capacity(COL_KEY_DIM);
            for _ in 0..COL_KEY_DIM { k.push(fast_randf(&mut state)); }
            column_keys.push(k);
        }
        let col_input_stride = (input_pop / COL_KEY_DIM).max(1);

        Self {
            query_weights,
            key_weights,
            gain: 1.0,
            sample_stride,
            input_samples,
            hidden_samples,
            column_keys,
            col_input_stride,
        }
    }

    /// Compress an input frame into a COL_KEY_DIM-dim vector by stride
    /// sampling. Same projection used by attention keys, so similarity is
    /// computed in a consistent space.
    pub fn compress_input(&self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(COL_KEY_DIM);
        for i in 0..COL_KEY_DIM {
            let idx = i * self.col_input_stride;
            out.push(input.get(idx).copied().unwrap_or(0.0));
        }
        out
    }

    /// Per-column gain in [0.6, 1.6], from cosine-like similarity between
    /// compressed input and each column's learned key. Returns one gain
    /// per column. Columns whose keys match the current frame are amplified;
    /// orthogonal columns are mildly suppressed.
    pub fn column_gains(&self, compressed_input: &[f32]) -> Vec<f32> {
        self.column_keys.iter().map(|key| {
            let dot: f32 = key.iter().zip(compressed_input.iter())
                .map(|(k, x)| k * x).sum();
            // Sigmoid-like map with center at 0 → gain 1.0
            let g = 1.0 + dot.tanh() * 0.6;
            g.clamp(0.6, 1.6)
        }).collect()
    }

    /// Hebbian update: when a column's mean firing activity is above
    /// baseline, its attention key drifts toward the compressed input.
    /// `column_activities` is per-column mean firing rate (length =
    /// num_columns), in [0, 1].
    pub fn learn_column_keys(
        &mut self,
        compressed_input: &[f32],
        column_activities: &[f32],
        lr: f32,
    ) {
        let baseline = column_activities.iter().sum::<f32>()
            / column_activities.len().max(1) as f32;
        for (c, key) in self.column_keys.iter_mut().enumerate() {
            let activity = column_activities.get(c).copied().unwrap_or(0.0);
            let drive = (activity - baseline).max(0.0);
            if drive < 1e-3 { continue; }
            let step = lr * drive;
            for (k, x) in key.iter_mut().zip(compressed_input.iter()) {
                *k += step * (x - *k);   // exponential moving average toward input
            }
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
