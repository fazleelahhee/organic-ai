use serde::{Deserialize, Serialize};

/// Tracks interaction patterns across sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMemory {
    pub total_sessions: u32,
    pub total_ticks: u64,
    pub total_messages_received: u32,
    pub interaction_patterns: Vec<InteractionRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub tick: u64,
    pub message_hash: u32,
    pub organism_response_energy: f32,
}

impl SessionMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_session_start(&mut self) {
        self.total_sessions += 1;
    }

    pub fn record_interaction(&mut self, tick: u64, message: &str, response_energy: f32) {
        let hash = message.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        self.interaction_patterns.push(InteractionRecord {
            tick,
            message_hash: hash,
            organism_response_energy: response_energy,
        });
        self.total_messages_received += 1;

        // Keep only last 1000 interactions
        if self.interaction_patterns.len() > 1000 {
            self.interaction_patterns.drain(..self.interaction_patterns.len() - 1000);
        }
    }

    pub fn interaction_count(&self) -> usize {
        self.interaction_patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_memory_records() {
        let mut mem = SessionMemory::new();
        mem.record_session_start();
        mem.record_interaction(100, "hello", 5.0);
        assert_eq!(mem.total_sessions, 1);
        assert_eq!(mem.total_messages_received, 1);
        assert_eq!(mem.interaction_count(), 1);
    }

    #[test]
    fn test_interaction_limit() {
        let mut mem = SessionMemory::new();
        for i in 0..1500 {
            mem.record_interaction(i as u64, &format!("msg{}", i), 1.0);
        }
        assert_eq!(mem.interaction_count(), 1000); // capped
    }
}
