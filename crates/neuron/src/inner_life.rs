/// Inner life — the brain thinks for itself when nobody is asking.
///
/// Like a human mind that never stops:
/// - Daydreams (random recall → chain → new connections)
/// - Rehearses (re-activates recent memories to strengthen them)
/// - Wonders (chains facts and discovers new relationships)
/// - Notices (detects patterns across stored knowledge)
///
/// All through existing Hebbian weights. No new mechanisms.
/// The curiosity drive triggers thinking when information gain is low.

use crate::memory::AttractorMemory;
use crate::thinking::{chain_recall, ConversationContext};
use serde::{Deserialize, Serialize};

/// A thought — something the brain discovered on its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub seed: String,      // what triggered the thought
    pub chain: Vec<String>, // the chain of associations
    pub insight: String,    // what it discovered
    pub tick: u64,
}

/// The brain's inner life — runs continuously in background.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerLife {
    /// Seeds for daydreaming — recent cues the brain has processed
    recent_cues: Vec<String>,
    max_recent: usize,
    /// Thoughts the brain has had on its own
    pub thoughts: Vec<Thought>,
    max_thoughts: usize,
    /// Curiosity level — high when not enough new input
    boredom: f32,
    /// Is the brain currently busy processing a query?
    busy: bool,
    /// Counter for tracking
    pub total_thoughts: u64,
}

impl InnerLife {
    pub fn new() -> Self {
        Self {
            recent_cues: Vec::new(),
            max_recent: 50,
            thoughts: Vec::new(),
            max_thoughts: 100,
            boredom: 0.0,
            busy: false,
            total_thoughts: 0,
        }
    }

    /// The brain is now busy (processing a query).
    pub fn set_busy(&mut self) { self.busy = true; }

    /// The brain is now free.
    pub fn set_free(&mut self) { self.busy = false; }

    /// Record that someone asked the brain something.
    pub fn record_interaction(&mut self, cue: &str) {
        self.recent_cues.push(cue.to_string());
        if self.recent_cues.len() > self.max_recent {
            self.recent_cues.remove(0);
        }
        self.boredom -= 0.3;
        if self.boredom < 0.0 { self.boredom = 0.0; }
    }

    /// Increase boredom over time — no input = brain gets restless.
    pub fn tick_boredom(&mut self) {
        self.boredom += 0.001;
        if self.boredom > 1.0 { self.boredom = 1.0; }
    }

    /// Should the brain think right now?
    /// YES when: not busy AND bored enough AND has something to think about.
    /// NO fixed timer. The brain decides based on its own state.
    pub fn should_think(&self) -> bool {
        !self.busy && self.boredom > 0.3 && !self.recent_cues.is_empty()
    }

    /// The brain thinks for itself.
    /// Picks a random recent cue, chains recalls, discovers connections.
    /// Stores new insights back in memory.
    pub fn daydream(&mut self, memory: &mut AttractorMemory, tick: u64) -> Option<Thought> {
        if self.recent_cues.is_empty() { return None; }

        // Pick a seed — use tick as pseudo-random index
        let seed_idx = (tick as usize) % self.recent_cues.len();
        let seed = self.recent_cues[seed_idx].clone();

        // Chain recall — follow associations
        let chain = chain_recall(memory, &seed, 4);

        if chain.len() >= 2 {
            // The brain discovered a connection!
            // Combine first and last elements as a new insight
            let insight = format!("{} connects to {}", chain[0], chain[chain.len() - 1]);

            // Store the insight back in memory — the brain teaches itself
            memory.store(&seed, &insight);

            let thought = Thought {
                seed: seed.clone(),
                chain: chain.clone(),
                insight: insight.clone(),
                tick,
            };

            self.thoughts.push(thought.clone());
            if self.thoughts.len() > self.max_thoughts {
                self.thoughts.remove(0);
            }
            self.total_thoughts += 1;
            self.boredom -= 0.2;

            Some(thought)
        } else {
            // Couldn't chain — rehearse the seed instead
            // Re-activating a memory strengthens it (Hebbian reinforcement)
            let recalled = memory.recall(&seed);
            if !recalled.trim().is_empty() {
                memory.store(&seed, &recalled); // re-store = reinforce
            }
            self.boredom -= 0.05;
            None
        }
    }

    /// Get recent thoughts for display.
    pub fn recent_thoughts(&self, count: usize) -> &[Thought] {
        let start = if self.thoughts.len() > count { self.thoughts.len() - count } else { 0 };
        &self.thoughts[start..]
    }

    pub fn boredom_level(&self) -> f32 {
        self.boredom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_life_creation() {
        let il = InnerLife::new();
        assert_eq!(il.total_thoughts, 0);
        assert_eq!(il.boredom_level(), 0.0);
    }

    #[test]
    fn test_boredom_increases() {
        let mut il = InnerLife::new();
        for _ in 0..100 { il.tick_boredom(); }
        assert!(il.boredom_level() > 0.0);
    }

    #[test]
    fn test_interaction_reduces_boredom() {
        let mut il = InnerLife::new();
        for _ in 0..100 { il.tick_boredom(); }
        let before = il.boredom_level();
        il.record_interaction("hello");
        assert!(il.boredom_level() < before);
    }

    #[test]
    fn test_daydream_with_knowledge() {
        let mut il = InnerLife::new();
        let mut mem = AttractorMemory::new();
        mem.store("dog", "animal");
        mem.store("animal", "living thing");
        mem.store("living thing", "biology");

        il.record_interaction("dog");
        let thought = il.daydream(&mut mem, 100);

        // Should have chained: dog → animal → living thing → biology
        if let Some(t) = thought {
            assert!(!t.chain.is_empty());
            assert!(il.total_thoughts > 0);
        }
    }

    #[test]
    fn test_no_daydream_without_cues() {
        let mut il = InnerLife::new();
        let mut mem = AttractorMemory::new();
        let thought = il.daydream(&mut mem, 100);
        assert!(thought.is_none());
    }

    #[test]
    fn test_should_think_when_bored_and_free() {
        let mut il = InnerLife::new();
        il.record_interaction("test");
        // Not bored yet
        assert!(!il.should_think());
        // Get bored
        for _ in 0..500 { il.tick_boredom(); }
        assert!(il.should_think());
        // Now busy
        il.set_busy();
        assert!(!il.should_think());
        // Free again
        il.set_free();
        assert!(il.should_think());
    }
}
