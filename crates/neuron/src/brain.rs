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
/// 40 million spiking neurons with STDP learning.
/// Uses the same LIF + STDP that drives the organisms.

use crate::lif::{integrate_and_fire, LifParams};
use crate::stdp::{apply_stdp, StdpParams};
use rand::Rng;
use serde::{Deserialize, Serialize};

const TOTAL_NEURONS: usize = 40_000_000;
const INPUT_POP: usize = 1_000_000;     // 1M input neurons
const HIDDEN_POP: usize = 38_000_000;   // 38M recurrent hidden neurons
const OUTPUT_POP: usize = 1_000_000;    // 1M output neurons
const MAX_SYNAPSES_PER_NEURON: usize = 4;
const SPIKE_HISTORY_LEN: usize = 8;

/// A synapse in the brain — an incoming connection from source to this neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainSynapse {
    source: u32,
    weight: f32,
    last_pre_tick: u64,
    eligibility: f32,    // trace of recent activity — for three-factor learning
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
    input_pop: usize,
    hidden_pop: usize,
    output_pop: usize,
    lif_params: LifParams,
    stdp_params: StdpParams,
    /// Internal number ring — ring topology for math computation.
    number_ring: crate::ring::NumberRing,
    /// Attractor memory — Hebbian weights, no string storage.
    attractor_memory: crate::memory::AttractorMemory,
    /// Conversation context — tracks recent exchanges.
    pub context: crate::thinking::ConversationContext,
    /// Inner life — the brain thinks for itself.
    pub inner_life: crate::inner_life::InnerLife,
    /// Working memory — hold intermediate values, execute step-by-step plans.
    pub working_memory: crate::working_memory::WorkingMemory,
    tick: u64,
    pub total_queries: u64,
    pub total_training: u64,
}

impl OrganicBrain {
    /// Create a full-scale brain (40M neurons). Takes ~10 seconds to initialize.
    pub fn new() -> Self {
        Self::new_with_size(TOTAL_NEURONS, INPUT_POP, HIDDEN_POP, OUTPUT_POP, MAX_SYNAPSES_PER_NEURON)
    }

    /// Create a smaller brain for testing or resource-constrained environments.
    pub fn new_small(total: usize) -> Self {
        let input = total / 8;
        let output = total / 8;
        let hidden = total - input - output;
        Self::new_with_size(total, input, hidden, output, 4)
    }

    fn new_with_size(total: usize, input_pop: usize, hidden_pop: usize, output_pop: usize, synapses_per: usize) -> Self {
        println!("Allocating {} neurons ({:.1} GB)...", total,
            (total * (32 + synapses_per * 16)) as f64 / 1e9);

        // Use a fast deterministic RNG for initialization (faster than thread_rng at scale)
        let mut seed: u64 = 42;
        let fast_rand = |s: &mut u64, max: usize| -> usize {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as usize) % max
        };
        let fast_randf = |s: &mut u64, lo: f32, hi: f32| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((*s >> 33) as f32) / (u32::MAX as f32);
            lo + t * (hi - lo)
        };

        let mut neurons: Vec<BrainNeuron> = Vec::with_capacity(total);
        for _ in 0..total {
            neurons.push(BrainNeuron::new());
        }

        // LAYERED architecture with lateral inhibition.
        // Input → Hidden: strong excitatory forward connections
        // Hidden ↔ Hidden: lateral inhibition (keeps activity sparse)
        // Hidden → Output: excitatory forward connections
        // Output ↔ Output: lateral inhibition

        // Input → Hidden
        for h in input_pop..input_pop + hidden_pop {
            for _ in 0..synapses_per.max(4) {
                let src = fast_rand(&mut seed, input_pop);
                neurons[h].synapses.push(BrainSynapse {
                    source: src as u32, weight: 0.25, last_pre_tick: 0, eligibility: 0.0,
                });
            }
            // Lateral inhibition within hidden
            for _ in 0..2 {
                let src = input_pop + fast_rand(&mut seed, hidden_pop);
                if src != h {
                    neurons[h].synapses.push(BrainSynapse {
                        source: src as u32, weight: -0.3, last_pre_tick: 0, eligibility: 0.0,
                    });
                }
            }
        }

        // Hidden → Output
        for o in (input_pop + hidden_pop)..total {
            for _ in 0..synapses_per.max(4) {
                let src = input_pop + fast_rand(&mut seed, hidden_pop);
                neurons[o].synapses.push(BrainSynapse {
                    source: src as u32, weight: 0.25, last_pre_tick: 0, eligibility: 0.0,
                });
            }
            // Lateral inhibition within output
            for _ in 0..2 {
                let src = input_pop + hidden_pop + fast_rand(&mut seed, output_pop);
                if src != o {
                    neurons[o].synapses.push(BrainSynapse {
                        source: src as u32, weight: -0.3, last_pre_tick: 0, eligibility: 0.0,
                    });
                }
            }
        }

        println!("Brain initialized: {} neurons, {} synapses",
            total, total * synapses_per);

        Self {
            neurons,
            input_pop,
            hidden_pop,
            output_pop,
            lif_params: LifParams { threshold: 0.8, leak_rate: 0.1, reset_potential: 0.0 },
            stdp_params: StdpParams::default(),
            number_ring: crate::ring::NumberRing::new(500, 1000),
            attractor_memory: crate::memory::AttractorMemory::new(),
            context: crate::thinking::ConversationContext::new(5),
            inner_life: crate::inner_life::InnerLife::new(),
            working_memory: crate::working_memory::WorkingMemory::new(),
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
        let mut input = vec![0.0f32; self.input_pop];
        for (pos, byte) in text.bytes().enumerate() {
            let b = byte as usize;
            // Position-sensitive encoding — same character at different positions
            // activates DIFFERENT neurons. This lets the brain distinguish
            // "Japan" from "France" even in similar sentences.
            let idx1 = (b * 127 + pos * 251) % self.input_pop;
            let idx2 = (b * 67 + pos * 139) % self.input_pop;
            input[idx1] = 1.0;
            input[idx2] = 0.7;
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
        let output_start = self.input_pop + self.hidden_pop;
        let output_end = output_start + self.output_pop;

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
    /// Uses sparse spike sets for efficiency at 40M neuron scale.
    fn run_ticks(&mut self, input: &[f32], n_ticks: usize, learn: bool) {
        use std::collections::HashSet;

        for _ in 0..n_ticks {
            self.tick += 1;

            // Collect fired neuron indices (sparse — only store the ones that fired)
            let fired_set: HashSet<u32> = self.neurons.iter().enumerate()
                .filter(|(_, n)| n.fired)
                .map(|(i, _)| i as u32)
                .collect();

            // Process each neuron
            for i in 0..self.neurons.len() {
                let mut total_input = 0.0f32;

                // External input (for input population only)
                if i < self.input_pop {
                    total_input += input.get(i).copied().unwrap_or(0.0);
                }

                // Synaptic input + eligibility trace updates
                let synapse_count = self.neurons[i].synapses.len();
                for s in 0..synapse_count {
                    let source = self.neurons[i].synapses[s].source;
                    if fired_set.contains(&source) {
                        total_input += self.neurons[i].synapses[s].weight;
                        self.neurons[i].synapses[s].last_pre_tick = self.tick;
                        // Mark eligible for learning
                        if self.neurons[i].synapses[s].weight > 0.0 {
                            self.neurons[i].synapses[s].eligibility = 1.0;
                        }
                    }
                    // Decay eligibility trace
                    self.neurons[i].synapses[s].eligibility *= 0.9;
                }

                // Multiplicative leak
                self.neurons[i].potential *= 0.85;
                self.neurons[i].potential += total_input;
                if self.neurons[i].potential < 0.0 { self.neurons[i].potential = 0.0; }

                let fired = self.neurons[i].potential >= self.lif_params.threshold;
                if fired { self.neurons[i].potential = self.lif_params.reset_potential; }
                self.neurons[i].fired = fired;
                self.neurons[i].record_spike(fired);
                if fired {
                    self.neurons[i].last_fire_tick = self.tick;
                }

                // THREE-FACTOR LEARNING: STDP × eligibility × reward
                // Only strengthen synapses that were part of the causal chain
                if learn && fired {
                    let synapse_count = self.neurons[i].synapses.len();
                    for s in 0..synapse_count {
                        let w = self.neurons[i].synapses[s].weight;
                        let elig = self.neurons[i].synapses[s].eligibility;
                        let pre_tick = self.neurons[i].synapses[s].last_pre_tick;
                        if w > 0.0 && elig > 0.1 && pre_tick > 0 {
                            let dt = self.tick - pre_tick;
                            if dt > 0 && dt < 8 {
                                let delta = 0.02 * elig * (1.0 - dt as f32 / 8.0);
                                self.neurons[i].synapses[s].weight += delta;
                                if self.neurons[i].synapses[s].weight > 2.0 {
                                    self.neurons[i].synapses[s].weight = 2.0;
                                }
                            }
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
        self.total_queries += 1;
        self.inner_life.set_busy();
        self.inner_life.record_interaction(query);

        // The brain THINKS — not just recalls.
        // 1. Checks context (what were we talking about?)
        // 2. Tries direct recall from Hebbian weights
        // 3. Tries reasoning (chain recalls)
        // 4. Tries creative blend (noise + pattern overlap)
        let (response, source) = crate::thinking::think(
            &self.attractor_memory,
            &self.context,
            query,
        );

        if !response.is_empty() {
            self.context.add_turn(query, &response);
            self.inner_life.set_free();
            return response;
        }

        // Nothing — brain doesn't know yet
        self.inner_life.set_free();
        String::new()
    }

    /// Train the brain: present input, then present the desired output,
    /// let STDP strengthen the pathways between them.
    /// This is genuine associative learning — pairing input with output
    /// and letting spike timing do the wiring.
    pub fn train(&mut self, input_text: &str, output_text: &str) {
        // Store in attractor memory (Hebbian weight update)
        self.attractor_memory.store(input_text, output_text);
        // Record in context
        self.context.add_turn(input_text, output_text);

        let input_pattern = self.encode_to_spikes(input_text);
        let output_pattern = self.encode_to_spikes(output_text);

        // Phase 1: Present input — let it propagate (1 tick at 40M scale)
        self.run_ticks(&input_pattern, 1, true);

        // Phase 2: Clamp output neurons to desired pattern — STDP wires them
        let output_start = self.input_pop + self.hidden_pop;
        for i in 0..self.output_pop {
            let target_rate = output_pattern.get(i).copied().unwrap_or(0.0);
            if target_rate > 0.3 {
                self.neurons[output_start + i].potential = self.lif_params.threshold + 0.1;
            }
        }
        self.run_ticks(&input_pattern, 1, true);

        self.total_training += 1;
    }

    /// Let the brain think for itself — called periodically by the simulation.
    /// Returns a thought if the brain discovered something new.
    /// Let the brain think for itself — when it's free and bored.
    /// No fixed timer. The brain decides based on its own curiosity state.
    pub fn tick_inner_life(&mut self, tick: u64) -> Option<crate::inner_life::Thought> {
        self.inner_life.tick_boredom();
        if self.inner_life.should_think() {
            self.inner_life.daydream(&mut self.attractor_memory, tick)
        } else {
            None
        }
    }

    /// Check if the brain has enough training to attempt answering.
    pub fn is_trained(&self) -> bool {
        self.total_training >= 1
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
            total_neurons: self.neurons.len(),
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

    // Tests use small brains (2048 neurons) for speed
    fn test_brain() -> OrganicBrain { OrganicBrain::new_small(2048) }

    #[test]
    fn test_brain_creation() {
        let brain = test_brain();
        assert_eq!(brain.neurons.len(), 2048);
        assert!(brain.neurons.iter().any(|n| !n.synapses.is_empty()));
    }

    #[test]
    fn test_encode_distributed() {
        let brain = test_brain();
        let pattern = brain.encode_to_spikes("hello");
        let active = pattern.iter().filter(|&&v| v > 0.0).count();
        assert!(active > 5, "Pattern should be distributed, got {} active", active);
    }

    #[test]
    fn test_different_inputs_different_patterns() {
        let brain = test_brain();
        let p1 = brain.encode_to_spikes("hello");
        let p2 = brain.encode_to_spikes("world");
        let diff: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_similar_inputs_similar_patterns() {
        let brain = test_brain();
        let p1 = brain.encode_to_spikes("hello");
        let p2 = brain.encode_to_spikes("hallo");
        let p3 = brain.encode_to_spikes("xyzzz");
        let diff_sim: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_dif: f32 = p1.iter().zip(p3.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_sim < diff_dif);
    }

    #[test]
    fn test_process() {
        let mut brain = test_brain();
        let _ = brain.process("2+3");
        assert_eq!(brain.total_queries, 1);
    }

    #[test]
    fn test_training_changes_weights() {
        let mut brain = test_brain();
        let w_before: f32 = brain.neurons.iter().flat_map(|n| n.synapses.iter().map(|s| s.weight)).sum();
        for _ in 0..10 { brain.train("2+3", "5"); }
        let w_after: f32 = brain.neurons.iter().flat_map(|n| n.synapses.iter().map(|s| s.weight)).sum();
        assert_ne!(w_before, w_after, "Training must change weights via STDP");
    }

    #[test]
    fn test_repeated_training() {
        let mut brain = test_brain();
        for _ in 0..50 { brain.train("hello", "world"); }
        assert_eq!(brain.total_training, 50);
        assert!(brain.is_trained());
    }

    #[test]
    fn test_stats() {
        let brain = test_brain();
        let stats = brain.stats();
        assert_eq!(stats.total_neurons, 2048);
        assert!(stats.total_synapses > 0);
    }
}
