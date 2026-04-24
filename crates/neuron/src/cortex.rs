/// Cortex — a larger neural network for abstract concept learning.
///
/// Unlike the small organism neural networks (10-30 cells), the cortex
/// has hundreds of neurons organized into layers that can learn abstract
/// concepts like numbers and arithmetic through experience.
///
/// Learning is purely organic:
/// - Numbers become distinct spike patterns through exposure
/// - Operations become synaptic pathways through STDP
/// - Generalization emerges from pattern overlap
/// - No hardcoded math — everything is learned

use rand::Rng;
use serde::{Deserialize, Serialize};

const CORTEX_SIZE: usize = 256;     // total neurons
const INPUT_SIZE: usize = 64;       // input layer neurons
const HIDDEN_SIZE: usize = 128;     // hidden layer neurons
const OUTPUT_SIZE: usize = 64;      // output layer neurons
const THRESHOLD: f32 = 0.8;
const LEAK: f32 = 0.1;
const LEARNING_RATE: f32 = 0.05;
const WEIGHT_DECAY: f32 = 0.001;

/// A single cortex neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Neuron {
    potential: f32,
    fired: bool,
    last_fire_step: Option<u32>,
}

impl Neuron {
    fn new() -> Self {
        Self { potential: 0.0, fired: false, last_fire_step: None }
    }

    fn integrate(&mut self, input: f32, step: u32) -> bool {
        self.potential -= LEAK;
        if self.potential < 0.0 { self.potential = 0.0; }
        self.potential += input;

        if self.potential >= THRESHOLD {
            self.potential = 0.0;
            self.fired = true;
            self.last_fire_step = Some(step);
            true
        } else {
            self.fired = false;
            false
        }
    }
}

/// The cortex — learns concepts through experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cortex {
    neurons: Vec<Neuron>,
    /// Weights from input → hidden
    weights_ih: Vec<Vec<f32>>,
    /// Weights from hidden → output
    weights_ho: Vec<Vec<f32>>,
    /// How many training examples seen
    pub experience_count: u64,
    /// Running accuracy on recent predictions
    pub accuracy: f32,
}

impl Cortex {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        let neurons: Vec<Neuron> = (0..CORTEX_SIZE).map(|_| Neuron::new()).collect();

        // Random small weights
        let weights_ih: Vec<Vec<f32>> = (0..INPUT_SIZE)
            .map(|_| (0..HIDDEN_SIZE).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let weights_ho: Vec<Vec<f32>> = (0..HIDDEN_SIZE)
            .map(|_| (0..OUTPUT_SIZE).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        Self {
            neurons,
            weights_ih,
            weights_ho,
            experience_count: 0,
            accuracy: 0.0,
        }
    }

    /// Encode text into a spike pattern across input neurons.
    /// Each character influences multiple neurons (distributed representation).
    fn encode_input(&self, text: &str) -> Vec<f32> {
        let mut pattern = vec![0.0f32; INPUT_SIZE];
        for (i, byte) in text.bytes().enumerate() {
            // Hash each character to multiple neuron positions
            let b = byte as usize;
            pattern[b % INPUT_SIZE] += 1.0;
            pattern[(b * 7 + i * 13) % INPUT_SIZE] += 0.5;
            pattern[(b * 31 + i * 3) % INPUT_SIZE] += 0.3;
        }
        // Normalize
        let max = pattern.iter().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            for p in &mut pattern { *p /= max; }
        }
        pattern
    }

    /// Decode output spike pattern back to text.
    fn decode_output(&self, pattern: &[f32]) -> String {
        // Find the strongest activations and map them to characters
        let mut indexed: Vec<(usize, f32)> = pattern.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // The top activations encode the answer
        // For numbers: the index of the strongest neuron maps to a digit
        let top = &indexed[..indexed.len().min(10)];

        // Simple decode: strongest neuron index mod 100 gives a number
        if let Some(&(idx, strength)) = top.first() {
            if strength > 0.3 {
                return format!("{}", idx % 100);
            }
        }

        String::new()
    }

    /// Run the cortex forward: input pattern → hidden spikes → output pattern.
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let step = self.experience_count as u32;

        // Input → Hidden
        let mut hidden_input = vec![0.0f32; HIDDEN_SIZE];
        for (i, &inp) in input.iter().enumerate() {
            if inp > 0.1 {
                for (h, weight) in self.weights_ih[i].iter().enumerate() {
                    hidden_input[h] += inp * weight;
                }
            }
        }

        // Hidden neurons fire
        let mut hidden_output = vec![0.0f32; HIDDEN_SIZE];
        for (h, inp) in hidden_input.iter().enumerate() {
            let neuron_idx = INPUT_SIZE + h;
            if neuron_idx < self.neurons.len() {
                if self.neurons[neuron_idx].integrate(*inp, step) {
                    hidden_output[h] = 1.0;
                }
            }
        }

        // Hidden → Output
        let mut output = vec![0.0f32; OUTPUT_SIZE];
        for (h, &ho) in hidden_output.iter().enumerate() {
            if ho > 0.5 {
                for (o, weight) in self.weights_ho[h].iter().enumerate() {
                    output[o] += ho * weight;
                }
            }
        }

        // Output neurons fire
        for (o, val) in output.iter_mut().enumerate() {
            let neuron_idx = INPUT_SIZE + HIDDEN_SIZE + o;
            if neuron_idx < self.neurons.len() {
                if self.neurons[neuron_idx].integrate(*val, step) {
                    *val = 1.0;
                } else {
                    *val = self.neurons[neuron_idx].potential;
                }
            }
        }

        output
    }

    /// Learn from an example: given input text and expected output text.
    /// Uses a biologically-plausible Hebbian/STDP-like rule:
    /// If input neuron and output neuron should both be active → strengthen
    /// If input neuron active but output neuron shouldn't be → weaken
    pub fn learn(&mut self, input_text: &str, expected_output: &str) {
        let input_pattern = self.encode_input(input_text);
        let target_pattern = self.encode_input(expected_output);

        // Forward pass
        let actual_output = self.forward(&input_pattern);

        // Compute error signal
        let mut error = vec![0.0f32; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE.min(target_pattern.len()) {
            error[i] = target_pattern[i] - actual_output[i];
        }

        // Hebbian-like learning on hidden → output weights
        // "Neurons that fire together wire together"
        for h in 0..HIDDEN_SIZE {
            let h_neuron_idx = INPUT_SIZE + h;
            let h_active = h_neuron_idx < self.neurons.len()
                && self.neurons[h_neuron_idx].fired;

            if h_active {
                for o in 0..OUTPUT_SIZE {
                    // Strengthen connections that would produce the right output
                    self.weights_ho[h][o] += LEARNING_RATE * error[o];
                    // Weight decay (prevents explosion)
                    self.weights_ho[h][o] *= 1.0 - WEIGHT_DECAY;
                    self.weights_ho[h][o] = self.weights_ho[h][o].clamp(-2.0, 2.0);
                }
            }
        }

        // Hebbian-like learning on input → hidden weights
        for i in 0..INPUT_SIZE {
            if input_pattern[i] > 0.1 {
                for h in 0..HIDDEN_SIZE {
                    let h_neuron_idx = INPUT_SIZE + h;
                    let h_active = h_neuron_idx < self.neurons.len()
                        && self.neurons[h_neuron_idx].fired;

                    if h_active {
                        // Reinforce active pathways that contributed to correct output
                        let output_error_sum: f32 = self.weights_ho[h].iter()
                            .zip(error.iter())
                            .map(|(w, e)| w * e)
                            .sum();

                        self.weights_ih[i][h] += LEARNING_RATE * input_pattern[i] * output_error_sum * 0.1;
                        self.weights_ih[i][h] *= 1.0 - WEIGHT_DECAY;
                        self.weights_ih[i][h] = self.weights_ih[i][h].clamp(-2.0, 2.0);
                    }
                }
            }
        }

        // Track accuracy
        let correct = target_pattern.iter().zip(actual_output.iter())
            .map(|(t, a)| if (t - a).abs() < 0.3 { 1.0 } else { 0.0 })
            .sum::<f32>() / OUTPUT_SIZE as f32;
        self.accuracy = self.accuracy * 0.95 + correct * 0.05; // exponential moving average

        self.experience_count += 1;
    }

    /// Try to answer a question using only internal neural processing.
    /// Returns Some(answer) if confident, None if unsure.
    pub fn try_answer(&mut self, question: &str) -> Option<String> {
        let input = self.encode_input(question);
        let output = self.forward(&input);

        // Check confidence — is there a strong activation?
        let max_activation = output.iter().cloned().fold(0.0f32, f32::max);

        if max_activation > 0.5 && self.experience_count > 10 {
            let answer = self.decode_output(&output);
            if !answer.is_empty() {
                return Some(answer);
            }
        }

        None // not confident enough
    }

    /// Get the number of training examples the cortex has processed.
    pub fn experience(&self) -> u64 {
        self.experience_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortex_creation() {
        let cortex = Cortex::new();
        assert_eq!(cortex.experience_count, 0);
        assert_eq!(cortex.neurons.len(), CORTEX_SIZE);
    }

    #[test]
    fn test_encode_produces_pattern() {
        let cortex = Cortex::new();
        let pattern = cortex.encode_input("hello");
        assert_eq!(pattern.len(), INPUT_SIZE);
        assert!(pattern.iter().any(|&p| p > 0.0));
    }

    #[test]
    fn test_different_inputs_different_patterns() {
        let cortex = Cortex::new();
        let p1 = cortex.encode_input("2+3");
        let p2 = cortex.encode_input("7+1");
        // Patterns should differ
        let diff: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "Different inputs should produce different patterns");
    }

    #[test]
    fn test_learning_changes_weights() {
        let mut cortex = Cortex::new();
        let w_before: f32 = cortex.weights_ho.iter()
            .flat_map(|row| row.iter())
            .sum();

        for _ in 0..50 {
            cortex.learn("2+3", "5");
        }

        let w_after: f32 = cortex.weights_ho.iter()
            .flat_map(|row| row.iter())
            .sum();

        assert_ne!(w_before, w_after, "Learning should modify weights");
    }

    #[test]
    fn test_experience_counter() {
        let mut cortex = Cortex::new();
        cortex.learn("1+1", "2");
        cortex.learn("2+2", "4");
        assert_eq!(cortex.experience_count, 2);
    }

    #[test]
    fn test_forward_produces_output() {
        let mut cortex = Cortex::new();
        let input = cortex.encode_input("test");
        let output = cortex.forward(&input);
        assert_eq!(output.len(), OUTPUT_SIZE);
    }
}
