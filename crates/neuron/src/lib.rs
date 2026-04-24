// Genuine neural components only — no parsers, no dictionaries, no hardcoded logic.
pub mod lif;        // Leaky Integrate-and-Fire — real spiking neuron model
pub mod stdp;       // Spike-Timing-Dependent Plasticity — real learning rule
pub mod curiosity;  // Information gain — real intrinsic motivation
pub mod spike;      // Spike propagation — real neural communication
pub mod brain;      // OrganicBrain — 40M spiking neurons with STDP
pub mod ring;       // NumberRing — ring attractor for organic math
