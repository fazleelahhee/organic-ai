use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple associative memory — stores signal patterns keyed by an address signal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMemory {
    storage: HashMap<u32, Vec<f32>>,
    max_entries: usize,
}

impl ToolMemory {
    pub fn new(max_entries: usize) -> Self {
        Self { storage: HashMap::new(), max_entries }
    }

    /// Store a pattern at an address. Returns true if stored.
    pub fn store(&mut self, address: f32, pattern: Vec<f32>) -> bool {
        let key = (address * 1000000.0) as u32;
        if self.storage.len() >= self.max_entries && !self.storage.contains_key(&key) {
            return false;
        }
        self.storage.insert(key, pattern);
        true
    }

    /// Retrieve a pattern by address. Returns empty vec if not found.
    pub fn retrieve(&self, address: f32) -> Vec<f32> {
        let key = (address * 1000000.0) as u32;
        self.storage.get(&key).cloned().unwrap_or_default()
    }

    pub fn entry_count(&self) -> usize {
        self.storage.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = ToolMemory::new(10);
        assert!(mem.store(0.5, vec![1.0, 2.0, 3.0]));
        let result = mem.retrieve(0.5);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_retrieve_missing() {
        let mem = ToolMemory::new(10);
        assert!(mem.retrieve(0.5).is_empty());
    }

    #[test]
    fn test_max_entries() {
        let mut mem = ToolMemory::new(2);
        mem.store(0.1, vec![1.0]);
        mem.store(0.2, vec![2.0]);
        assert!(!mem.store(0.3, vec![3.0])); // full
    }
}
