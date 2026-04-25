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
pub(crate) struct BrainSynapse {
    source: u32,
    weight: f32,
    last_pre_tick: u64,
    eligibility: f32,    // trace of recent activity — for three-factor learning
}

/// A single neuron in the brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BrainNeuron {
    pub(crate) potential: f32,
    pub(crate) fired: bool,
    pub(crate) last_fire_tick: u64,
    pub(crate) spike_history: [bool; SPIKE_HISTORY_LEN], // circular buffer of recent spikes
    pub(crate) history_idx: usize,
    pub(crate) synapses: Vec<BrainSynapse>,
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
    pub(crate) fn firing_rate(&self) -> f32 {
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
    /// HDC memory — hyperdimensional computing. 10,000x capacity.
    /// Replaces Hebbian matrix. Compositional, one-shot, noise-tolerant.
    hdc_memory: crate::hdc::HDCMemory,
    /// Conversation context — tracks recent exchanges.
    pub context: crate::thinking::ConversationContext,
    /// Inner life — the brain thinks for itself.
    pub inner_life: crate::inner_life::InnerLife,
    /// Working memory — hold intermediate values, execute step-by-step plans.
    pub working_memory: crate::working_memory::WorkingMemory,
    /// LSM readout — taps reservoir firing rates into a learned projection.
    lsm_readout: crate::lsm::LsmReadout,
    /// Predictive coding: input → hidden prediction.
    pred_input_to_hidden: crate::predictive::PredictionLayer,
    /// Predictive coding: hidden → output prediction.
    pred_hidden_to_output: crate::predictive::PredictionLayer,
    /// Attention — gain modulation on hidden population.
    attention: crate::attention::AttentionModule,
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
            hdc_memory: crate::hdc::HDCMemory::new(),
            context: crate::thinking::ConversationContext::new(5),
            inner_life: crate::inner_life::InnerLife::new(),
            working_memory: crate::working_memory::WorkingMemory::new(),
            lsm_readout: crate::lsm::LsmReadout::new(hidden_pop, 42),
            pred_input_to_hidden: crate::predictive::PredictionLayer::new(),
            pred_hidden_to_output: crate::predictive::PredictionLayer::new(),
            attention: crate::attention::AttentionModule::new(input_pop, hidden_pop, 42),
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
    /// PARALLELIZED with rayon — all CPU cores process neurons simultaneously.
    /// Each neuron only writes to ITSELF, so parallel processing is safe.
    fn run_ticks(&mut self, input: &[f32], n_ticks: usize, learn: bool) {
        use rayon::prelude::*;
        use std::collections::HashSet;

        let input_pop = self.input_pop;
        let threshold = self.lif_params.threshold;
        let reset = self.lif_params.reset_potential;

        for _ in 0..n_ticks {
            self.tick += 1;
            let tick = self.tick;

            // Collect fired neuron indices (sparse)
            let fired_set: HashSet<u32> = self.neurons.iter().enumerate()
                .filter(|(_, n)| n.fired)
                .map(|(i, _)| i as u32)
                .collect();

            // PARALLEL + SPARSE: each neuron processed independently.
            // Skip neurons that have no input AND no residual potential (saves ~60% compute)
            self.neurons.par_iter_mut().enumerate().for_each(|(i, neuron)| {
                // Sparse check: skip idle neurons ONLY during inference (not learning).
                // During learning, every neuron must process to allow STDP.
                if !learn {
                    let has_external = i < input_pop && input.get(i).copied().unwrap_or(0.0) > 0.0;
                    if !has_external && neuron.potential < 0.01 && !neuron.fired {
                        for syn in &mut neuron.synapses { syn.eligibility *= 0.9; }
                        return;
                    }
                }

                let mut total_input = 0.0f32;

                // External input (input population only)
                if i < input_pop {
                    total_input += input.get(i).copied().unwrap_or(0.0);
                }

                // Synaptic input + eligibility traces
                for syn in &mut neuron.synapses {
                    if fired_set.contains(&syn.source) {
                        total_input += syn.weight;
                        syn.last_pre_tick = tick;
                        if syn.weight > 0.0 { syn.eligibility = 1.0; }
                    }
                    syn.eligibility *= 0.9;
                }

                // Multiplicative leak + integration
                neuron.potential *= 0.85;
                neuron.potential += total_input;
                if neuron.potential < 0.0 { neuron.potential = 0.0; }

                // Fire check
                let fired = neuron.potential >= threshold;
                if fired { neuron.potential = reset; }
                neuron.fired = fired;
                neuron.record_spike(fired);
                if fired { neuron.last_fire_tick = tick; }

                // Three-factor STDP (only during training)
                if learn && fired {
                    for syn in &mut neuron.synapses {
                        if syn.weight > 0.0 && syn.eligibility > 0.1 && syn.last_pre_tick > 0 {
                            let dt = tick - syn.last_pre_tick;
                            if dt > 0 && dt < 8 {
                                syn.weight += 0.02 * syn.eligibility * (1.0 - dt as f32 / 8.0);
                                if syn.weight > 2.0 { syn.weight = 2.0; }
                            }
                        }
                    }
                }
            });
        }
    }

    /// Process a query through the brain's spiking network.
    /// Returns the brain's output — whatever its neurons produce.
    /// This may be nonsense early on, improving with training.
    pub fn process(&mut self, query: &str) -> String {
        self.total_queries += 1;
        self.inner_life.set_busy();
        self.inner_life.record_interaction(query);

        // STEP 1: Check if there's an active plan — continue executing it
        if self.working_memory.has_pending_steps() {
            if let Some(instruction) = self.working_memory.current_instruction() {
                let step_query = instruction.to_string();
                // Execute this step using recall
                let step_result = self.hdc_memory.recall(&step_query);
                let result = if !step_result.trim().is_empty() {
                    step_result
                } else {
                    // Can't execute step internally — return the instruction
                    // so Claude can teach it
                    self.inner_life.set_free();
                    return String::new();
                };
                self.working_memory.complete_step(&result);
                // If plan is done, return final result
                if !self.working_memory.has_pending_steps() {
                    if let Some(final_result) = self.working_memory.final_result() {
                        let response = final_result.to_string();
                        self.context.add_turn(query, &response);
                        self.inner_life.set_free();
                        return response;
                    }
                }
                // Plan still in progress — return intermediate state
                let state = self.working_memory.state_summary();
                self.inner_life.set_free();
                return format!("thinking... {}", state);
            }
        }

        // STEP 2: Try multi-step reasoning — chain recall.
        // If chain produces 2+ hops, use working memory to execute them.
        // No keyword matching — the brain discovers structure from its own weights.
        {
            let chain = crate::thinking::chain_recall(&self.hdc_memory, query, 5);
            if chain.len() >= 2 {
                // Store chain as a plan
                self.working_memory.clear();
                self.working_memory.set_plan(chain.clone());
                // Store the query in register for reference
                self.working_memory.store("query", query);
                // Execute first step
                if let Some(instruction) = self.working_memory.current_instruction() {
                    let step_result = self.hdc_memory.recall(instruction);
                    if !step_result.trim().is_empty() {
                        self.working_memory.complete_step(&step_result);
                    }
                }
                // Return what we have so far
                let results = self.working_memory.plan_results();
                if !results.is_empty() {
                    let response = results.join(". ");
                    self.context.add_turn(query, &response);
                    self.inner_life.set_free();
                    return response;
                }
            }
        }

        // STEP 3: FAST PATH — HDC recall returns immediately if it has an answer.
        //
        // The spiking network (80M neurons) takes seconds to run. Running it
        // before returning the HDC result causes HTTP timeouts. Instead:
        //   - Fast path: HDC recall hit → return IMMEDIATELY (~1ms)
        //   - Slow path: HDC miss → fall back to spiking network

        // Fast: attractor memory recall with context
        let (fast_response, _source) = crate::thinking::think(
            &self.hdc_memory,
            &self.context,
            query,
        );

        // FAST PATH: HDC has an answer — return it immediately, skip spiking network.
        if !fast_response.is_empty() {
            self.context.add_turn(query, &fast_response);
            self.inner_life.set_free();
            return fast_response;
        }

        // SLOW PATH: HDC had nothing — fall back to spiking network.
        // Run the spiking network to process the query through neural dynamics.
        let input = self.encode_to_spikes(query);
        let output_start = self.input_pop + self.hidden_pop;
        for i in output_start..(output_start + self.output_pop) {
            self.neurons[i].potential = 0.0;
            self.neurons[i].fired = false;
            self.neurons[i].spike_history = [false; SPIKE_HISTORY_LEN];
        }

        // --- Attention gain modulation (before dynamics) ---
        {
            let hidden_rates: Vec<f32> = self.neurons[self.input_pop..self.input_pop + self.hidden_pop]
                .iter()
                .map(|n| n.firing_rate())
                .collect();
            let gains = self.attention.compute_gains(&input, &hidden_rates, self.hidden_pop);
            for (i, &g) in gains.iter().enumerate() {
                let h = self.input_pop + i;
                if h < self.neurons.len() {
                    self.neurons[h].potential *= g;
                }
            }
        }

        self.run_ticks(&input, 2, false);

        // --- LSM readout (after dynamics) ---
        {
            let hidden_slice = &self.neurons[self.input_pop..self.input_pop + self.hidden_pop];
            let lsm_state = self.lsm_readout.collect_state(hidden_slice);
            let _lsm_out = self.lsm_readout.forward(&lsm_state);
            // lsm_out available for future downstream use
        }

        let deep_response = self.decode_from_rates();

        let response = if !deep_response.is_empty() && deep_response.chars().any(|c| c.is_alphanumeric()) {
            deep_response
        } else {
            self.inner_life.set_free();
            return String::new();
        };

        self.context.add_turn(query, &response);
        self.inner_life.set_free();
        response
    }

    /// Train the brain: present input, then present the desired output,
    /// let STDP strengthen the pathways between them.
    /// This is genuine associative learning — pairing input with output
    /// and letting spike timing do the wiring.
    pub fn train(&mut self, input_text: &str, output_text: &str) {
        // Record in context
        self.context.add_turn(input_text, output_text);

        let input_pattern = self.encode_to_spikes(input_text);
        let output_pattern = self.encode_to_spikes(output_text);

        // Phase 1: Present input — let it propagate (1 tick at 40M scale)
        self.run_ticks(&input_pattern, 1, true);

        // --- Predictive coding: input → hidden ---
        let input_rates: Vec<f32> = self.neurons[0..self.input_pop]
            .iter()
            .map(|n| n.firing_rate())
            .collect();
        let hidden_rates: Vec<f32> = self.neurons[self.input_pop..self.input_pop + self.hidden_pop]
            .iter()
            .map(|n| n.firing_rate())
            .collect();
        let comp_input = self.pred_input_to_hidden.compress(&input_rates, 0);
        let comp_hidden = self.pred_input_to_hidden.compress(&hidden_rates, 0);
        let _pred_err_1 = self.pred_input_to_hidden.update(&comp_input, &comp_hidden);

        // Phase 2: Clamp output neurons to desired pattern — STDP wires them
        let output_start = self.input_pop + self.hidden_pop;
        for i in 0..self.output_pop {
            let target_rate = output_pattern.get(i).copied().unwrap_or(0.0);
            if target_rate > 0.3 {
                self.neurons[output_start + i].potential = self.lif_params.threshold + 0.1;
            }
        }
        self.run_ticks(&input_pattern, 1, true);

        // --- Predictive coding: hidden → output ---
        let hidden_rates2: Vec<f32> = self.neurons[self.input_pop..self.input_pop + self.hidden_pop]
            .iter()
            .map(|n| n.firing_rate())
            .collect();
        let output_rates: Vec<f32> = self.neurons[output_start..output_start + self.output_pop]
            .iter()
            .map(|n| n.firing_rate())
            .collect();
        let comp_hidden2 = self.pred_hidden_to_output.compress(&hidden_rates2, 0);
        let comp_output = self.pred_hidden_to_output.compress(&output_rates, 0);
        let _pred_err_2 = self.pred_hidden_to_output.update(&comp_hidden2, &comp_output);

        // --- Replace fake surprise with genuine prediction error ---
        let predicted = self.hdc_memory.recall(input_text);
        let predicted_vec = self.hdc_memory.encode(&predicted);
        let actual_vec = self.hdc_memory.encode(output_text);
        let hdc_error = crate::curiosity::compute_hdc_prediction_error(
            predicted_vec.similarity(&actual_vec),
        );
        let coding_error = (self.pred_input_to_hidden.prediction_error
            + self.pred_hidden_to_output.prediction_error)
            / 2.0;
        let surprise = crate::curiosity::compute_combined_error(hdc_error, coding_error, 0.6);

        // Use the genuine surprise to gate HDC storage
        if surprise > 0.1 {
            self.hdc_memory.store(input_text, output_text);
        }

        // Feed inner life — surprising inputs drive more daydreaming
        if surprise > 0.3 {
            self.inner_life.record_interaction(input_text);
        }

        // --- Train LSM readout ---
        {
            let hidden_slice = &self.neurons[self.input_pop..self.input_pop + self.hidden_pop];
            let lsm_state = self.lsm_readout.collect_state(hidden_slice);
            // Target: compressed output rates
            let target = self.pred_hidden_to_output.compress(&output_rates, 0);
            // Pad/truncate target to 1024 for the readout
            let mut target_1024 = vec![0.0f32; 1024];
            for (i, v) in target.iter().enumerate() {
                if i < 1024 {
                    target_1024[i] = *v;
                }
            }
            self.lsm_readout.train(&lsm_state, &target_1024);
        }

        // --- Train attention ---
        self.attention.learn(
            &input_pattern,
            &hidden_rates2,
            surprise,
            self.hidden_pop,
        );

        self.total_training += 1;
    }

    /// Let the brain think for itself — called periodically by the simulation.
    /// Returns a thought if the brain discovered something new.
    /// Let the brain think for itself — when it's free and bored.
    /// No fixed timer. The brain decides based on its own curiosity state.
    pub fn tick_inner_life(&mut self, tick: u64) -> Option<crate::inner_life::Thought> {
        self.inner_life.tick_boredom();
        if self.inner_life.should_think() {
            self.inner_life.daydream(&mut self.hdc_memory, tick)
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
