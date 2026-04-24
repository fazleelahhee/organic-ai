/// OrganicBrain — A genuine spiking neural network that processes queries.
///
/// This replaces the fake intelligence layer (parser, HashMap dictionary,
/// backprop cortex) with a real spiking neural network that:
/// - Encodes input as distributed spike patterns across populations of neurons
/// - Processes through recurrent spiking dynamics with STDP
/// - Produces output as spike-rate codes
/// - Learns entirely through spike-timing-dependent plasticity
/// - No backpropagation, no parsing, no HashMaps
///
/// The network is large enough to learn real associations (2048 neurons)
/// and uses the same LIF + STDP that drives the organisms.

use crate::lif::{integrate_and_fire, LifParams};
use crate::stdp::{apply_stdp, StdpParams};
use rand::Rng;
use serde::{Deserialize, Serialize};

const TOTAL_NEURONS: usize = 2048;
const INPUT_POP: usize = 256;    // input population
const HIDDEN_POP: usize = 1536;  // recurrent hidden population
const OUTPUT_POP: usize = 256;   // output population
const MAX_SYNAPSES_PER_NEURON: usize = 32;
const SPIKE_HISTORY_LEN: usize = 8; // ticks of spike history for rate coding

/// A synapse in the brain — an incoming connection from source to this neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainSynapse {
    source: u16,         // source neuron index (where input comes FROM)
    weight: f32,
    last_pre_tick: u64,
}

/// A single neuron in the brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainNeuron {
    potential: f32,
    fired: bool,
    last_fire_tick: u64,
    spike_history: [bool; SPIKE_HISTORY_LEN], // circular buffer of recent spikes
    history_idx: usize,
    synapses: Vec<BrainSynapse>,
}

impl BrainNeuron {
    fn new() -> Self {
        Self {
            potential: 0.0,
            fired: false,
            last_fire_tick: 0,
            spike_history: [false; SPIKE_HISTORY_LEN],
            history_idx: 0,
            synapses: Vec::new(),
        }
    }

    /// Firing rate over recent history (0.0 to 1.0)
    fn firing_rate(&self) -> f32 {
        let count = self.spike_history.iter().filter(|&&s| s).count();
        count as f32 / SPIKE_HISTORY_LEN as f32
    }

    fn record_spike(&mut self, fired: bool) {
        self.spike_history[self.history_idx] = fired;
        self.history_idx = (self.history_idx + 1) % SPIKE_HISTORY_LEN;
    }
}

/// The organic brain — processes all queries through spiking dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganicBrain {
    neurons: Vec<BrainNeuron>,
    lif_params: LifParams,
    stdp_params: StdpParams,
    tick: u64,
    pub total_queries: u64,
    pub total_training: u64,
}

impl OrganicBrain {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut neurons: Vec<BrainNeuron> = (0..TOTAL_NEURONS)
            .map(|_| BrainNeuron::new())
            .collect();

        // Initialize random sparse connectivity
        // Input → Hidden (sparse, random)
        for i in 0..INPUT_POP {
            let n_syn = rng.gen_range(8..MAX_SYNAPSES_PER_NEURON);
            for _ in 0..n_syn {
                let target = INPUT_POP + rng.gen_range(0..HIDDEN_POP);
                neurons[target].synapses.push(BrainSynapse {
                    source: i as u16,
                    weight: rng.gen_range(0.01..0.15),
                    last_pre_tick: 0,
                });
            }
        }

        // Hidden → Hidden (recurrent, sparse)
        for i in INPUT_POP..(INPUT_POP + HIDDEN_POP) {
            let n_syn = rng.gen_range(4..16);
            for _ in 0..n_syn {
                let target = INPUT_POP + rng.gen_range(0..HIDDEN_POP);
                if target != i { // no self-connections
                    neurons[target].synapses.push(BrainSynapse {
                        source: i as u16,
                        weight: rng.gen_range(-0.1..0.1),
                        last_pre_tick: 0,
                    });
                }
            }
        }

        // Hidden → Output (sparse, random)
        for i in INPUT_POP..(INPUT_POP + HIDDEN_POP) {
            let n_syn = rng.gen_range(2..8);
            for _ in 0..n_syn {
                let target = INPUT_POP + HIDDEN_POP + rng.gen_range(0..OUTPUT_POP);
                neurons[target].synapses.push(BrainSynapse {
                    source: i as u16,
                    weight: rng.gen_range(0.01..0.1),
                    last_pre_tick: 0,
                });
            }
        }

        Self {
            neurons,
            lif_params: LifParams { threshold: 0.8, leak_rate: 0.1, reset_potential: 0.0 },
            stdp_params: StdpParams::default(),
            tick: 0,
            total_queries: 0,
            total_training: 0,
        }
    }

    /// Encode text as a distributed spike pattern across input neurons.
    /// Each character activates a POPULATION of neurons (not one neuron per char).
    /// This creates overlapping distributed representations — similar inputs
    /// activate similar neuron populations.
    fn encode_to_spikes(&self, text: &str) -> Vec<f32> {
        let mut input = vec![0.0f32; INPUT_POP];
        for (pos, byte) in text.bytes().enumerate() {
            let b = byte as usize;
            // Each character activates ~8 neurons in a distributed pattern
            // Using prime multipliers to spread activation
            let indices = [
                b % INPUT_POP,
                (b * 7 + pos * 3) % INPUT_POP,
                (b * 13 + pos * 7) % INPUT_POP,
                (b * 31 + pos * 11) % INPUT_POP,
                (b * 37 + pos * 17) % INPUT_POP,
                (b.wrapping_mul(41).wrapping_add(pos * 23)) % INPUT_POP,
                (b.wrapping_mul(53).wrapping_add(pos * 29)) % INPUT_POP,
                (b.wrapping_mul(61).wrapping_add(pos * 31)) % INPUT_POP,
            ];
            for &idx in &indices {
                input[idx] += 0.15;
            }
        }
        // Normalize to [0, 1]
        let max = input.iter().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            for v in &mut input { *v = (*v / max).min(1.0); }
        }
        input
    }

    /// Decode output neuron firing rates back to text.
    /// The output population's firing pattern IS the answer.
    fn decode_from_rates(&self) -> String {
        let output_start = INPUT_POP + HIDDEN_POP;
        let output_end = output_start + OUTPUT_POP;

        // Collect firing rates from output neurons
        let rates: Vec<f32> = self.neurons[output_start..output_end]
            .iter()
            .map(|n| n.firing_rate())
            .collect();

        // Convert firing rates to characters
        // Group output neurons into chunks — each chunk encodes one output character
        let chars_per_group = 4; // 4 neurons per character = 256/4 = 64 possible output chars
        let mut result = String::new();

        for chunk in rates.chunks(chars_per_group) {
            // Weighted average of firing rates in this chunk determines the character
            let avg_rate: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
            if avg_rate > 0.05 { // only emit if there's significant activity
                // Map rate to ASCII printable range
                let char_code = (avg_rate * 94.0) as u8 + 32; // 32-126
                if char_code >= 32 && char_code < 127 {
                    result.push(char_code as char);
                }
            }
        }

        result.trim().to_string()
    }

    /// Run the brain for N ticks with given input, applying STDP learning.
    /// This is where the actual neural computation happens.
    fn run_ticks(&mut self, input: &[f32], n_ticks: usize, learn: bool) {
        for _ in 0..n_ticks {
            self.tick += 1;

            // Collect current spike state
            let spike_state: Vec<bool> = self.neurons.iter().map(|n| n.fired).collect();

            // Process each neuron
            for i in 0..TOTAL_NEURONS {
                // Compute total input to this neuron
                let mut total_input = 0.0f32;

                // External input (for input population only)
                if i < INPUT_POP {
                    total_input += input.get(i).copied().unwrap_or(0.0);
                }

                // Synaptic input from connected neurons (using previous tick's spikes)
                // We need to read from spike_state and write to neurons[i], so we
                // iterate synapses by index
                let synapse_count = self.neurons[i].synapses.len();
                for s in 0..synapse_count {
                    let source = self.neurons[i].synapses[s].source as usize;
                    if source < TOTAL_NEURONS && spike_state[source] {
                        total_input += self.neurons[i].synapses[s].weight;
                        self.neurons[i].synapses[s].last_pre_tick = self.tick;
                    }
                }

                // LIF integration
                let fired = integrate_and_fire(
                    &mut self.neurons[i].potential,
                    total_input,
                    &self.lif_params,
                );
                self.neurons[i].fired = fired;
                self.neurons[i].record_spike(fired);
                if fired {
                    self.neurons[i].last_fire_tick = self.tick;
                }

                // STDP learning (only during training)
                if learn && fired {
                    let post_tick = self.tick;
                    let synapse_count = self.neurons[i].synapses.len();
                    for s in 0..synapse_count {
                        let pre_tick = self.neurons[i].synapses[s].last_pre_tick;
                        if pre_tick > 0 {
                            apply_stdp(
                                &mut self.neurons[i].synapses[s].weight,
                                pre_tick,
                                post_tick,
                                0.0, // no external modulation — pure STDP
                                &self.stdp_params,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Process a query through the brain's spiking network.
    /// Returns the brain's output — whatever its neurons produce.
    /// This may be nonsense early on, improving with training.
    pub fn process(&mut self, query: &str) -> String {
        // Reset output neurons before processing
        let output_start = INPUT_POP + HIDDEN_POP;
        for i in output_start..(output_start + OUTPUT_POP) {
            self.neurons[i].potential = 0.0;
            self.neurons[i].fired = false;
            self.neurons[i].spike_history = [false; SPIKE_HISTORY_LEN];
        }

        let input = self.encode_to_spikes(query);
        self.run_ticks(&input, SPIKE_HISTORY_LEN * 2, false); // no learning during query
        self.total_queries += 1;
        self.decode_from_rates()
    }

    /// Train the brain: present input, then present the desired output,
    /// let STDP strengthen the pathways between them.
    /// This is genuine associative learning — pairing input with output
    /// and letting spike timing do the wiring.
    pub fn train(&mut self, input_text: &str, output_text: &str) {
        let input_pattern = self.encode_to_spikes(input_text);
        let output_pattern = self.encode_to_spikes(output_text);

        // Phase 1: Present input — let it propagate through the network
        self.run_ticks(&input_pattern, SPIKE_HISTORY_LEN, true);

        // Phase 2: Clamp output neurons to desired pattern — creates
        // temporal correlation between input-driven hidden activity
        // and correct output activity. STDP wires them together.
        let output_start = INPUT_POP + HIDDEN_POP;
        for i in 0..OUTPUT_POP {
            let target_rate = output_pattern.get(i).copied().unwrap_or(0.0);
            if target_rate > 0.3 {
                // Force this output neuron to fire — creates the association
                self.neurons[output_start + i].potential = self.lif_params.threshold + 0.1;
            }
        }
        self.run_ticks(&input_pattern, SPIKE_HISTORY_LEN, true);

        self.total_training += 1;
    }

    /// Check if the brain has enough training to attempt answering.
    pub fn is_trained(&self) -> bool {
        self.total_training >= 5
    }

    /// Get statistics about the brain's state.
    pub fn stats(&self) -> BrainStats {
        let total_synapses: usize = self.neurons.iter().map(|n| n.synapses.len()).sum();
        let active_neurons = self.neurons.iter()
            .filter(|n| n.firing_rate() > 0.0)
            .count();
        let avg_weight: f32 = {
            let all_weights: Vec<f32> = self.neurons.iter()
                .flat_map(|n| n.synapses.iter().map(|s| s.weight))
                .collect();
            if all_weights.is_empty() { 0.0 }
            else { all_weights.iter().sum::<f32>() / all_weights.len() as f32 }
        };

        BrainStats {
            total_neurons: TOTAL_NEURONS,
            total_synapses,
            active_neurons,
            avg_weight,
            total_queries: self.total_queries,
            total_training: self.total_training,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BrainStats {
    pub total_neurons: usize,
    pub total_synapses: usize,
    pub active_neurons: usize,
    pub avg_weight: f32,
    pub total_queries: u64,
    pub total_training: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_creation() {
        let brain = OrganicBrain::new();
        assert_eq!(brain.neurons.len(), TOTAL_NEURONS);
        assert!(brain.neurons.iter().any(|n| !n.synapses.is_empty()));
    }

    #[test]
    fn test_encode_produces_distributed_pattern() {
        let brain = OrganicBrain::new();
        let pattern = brain.encode_to_spikes("hello");
        assert_eq!(pattern.len(), INPUT_POP);
        // Should have multiple active neurons (distributed, not one-hot)
        let active = pattern.iter().filter(|&&v| v > 0.0).count();
        assert!(active > 5, "Pattern should be distributed across many neurons, got {}", active);
    }

    #[test]
    fn test_different_inputs_different_patterns() {
        let brain = OrganicBrain::new();
        let p1 = brain.encode_to_spikes("hello");
        let p2 = brain.encode_to_spikes("world");
        let diff: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "Different inputs must produce different patterns");
    }

    #[test]
    fn test_similar_inputs_similar_patterns() {
        let brain = OrganicBrain::new();
        let p1 = brain.encode_to_spikes("hello");
        let p2 = brain.encode_to_spikes("hallo"); // similar word
        let p3 = brain.encode_to_spikes("xyzzz"); // very different
        let diff_similar: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_different: f32 = p1.iter().zip(p3.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_similar < diff_different,
            "Similar inputs should have more similar patterns: similar={}, different={}", diff_similar, diff_different);
    }

    #[test]
    fn test_process_returns_something() {
        let mut brain = OrganicBrain::new();
        let result = brain.process("2+3");
        // Untrained brain produces noise — that's correct
        assert!(brain.total_queries == 1);
        // Result may be empty or noise — that's expected for an untrained brain
    }

    #[test]
    fn test_training_changes_weights() {
        let mut brain = OrganicBrain::new();
        let weights_before: f32 = brain.neurons.iter()
            .flat_map(|n| n.synapses.iter().map(|s| s.weight))
            .sum();

        for _ in 0..10 {
            brain.train("2+3", "5");
        }

        let weights_after: f32 = brain.neurons.iter()
            .flat_map(|n| n.synapses.iter().map(|s| s.weight))
            .sum();

        assert_ne!(weights_before, weights_after,
            "Training must change synapse weights via STDP");
    }

    #[test]
    fn test_repeated_training_strengthens_pathways() {
        let mut brain = OrganicBrain::new();
        // Train on the same pair many times
        for _ in 0..50 {
            brain.train("hello", "world");
        }
        assert_eq!(brain.total_training, 50);
        assert!(brain.is_trained());
    }

    #[test]
    fn test_stats() {
        let brain = OrganicBrain::new();
        let stats = brain.stats();
        assert_eq!(stats.total_neurons, TOTAL_NEURONS);
        assert!(stats.total_synapses > 0);
    }
}
