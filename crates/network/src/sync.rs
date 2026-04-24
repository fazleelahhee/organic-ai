/// Sync — distributed learning synchronization.
///
/// When two nodes both learn from interactions, their synapse weights diverge.
/// This module merges weights so both nodes benefit from each other's learning.
///
/// Like two brain hemispheres sharing what they've learned:
/// - Each hemisphere processes different inputs
/// - The corpus callosum syncs relevant patterns
/// - Both hemispheres become smarter together

use serde::{Deserialize, Serialize};

/// Strategy for merging weights from a remote node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Average local and remote weights (balanced learning).
    Average,
    /// Take the larger weight (keep strongest associations).
    Maximum,
    /// Weighted average biased toward local (local node is primary).
    LocalBiased { local_weight: f32 },
}

impl Default for MergeStrategy {
    fn default() -> Self {
        MergeStrategy::LocalBiased { local_weight: 0.7 }
    }
}

/// Merge a remote weight with a local weight.
pub fn merge_weight(local: f32, remote: f32, strategy: MergeStrategy) -> f32 {
    match strategy {
        MergeStrategy::Average => (local + remote) / 2.0,
        MergeStrategy::Maximum => local.max(remote),
        MergeStrategy::LocalBiased { local_weight } => {
            local * local_weight + remote * (1.0 - local_weight)
        }
    }
}

/// Collect weight changes that should be shared with peers.
/// Only shares weights that changed significantly (saves bandwidth).
pub fn collect_changed_weights(
    current_weights: &[(u32, u16, f32)],
    previous_weights: &[(u32, u16, f32)],
    threshold: f32,
) -> Vec<(u32, u16, f32)> {
    let mut changed = Vec::new();
    for (curr, prev) in current_weights.iter().zip(previous_weights.iter()) {
        if (curr.2 - prev.2).abs() > threshold {
            changed.push(*curr);
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_average() {
        let merged = merge_weight(0.4, 0.8, MergeStrategy::Average);
        assert!((merged - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_merge_maximum() {
        let merged = merge_weight(0.4, 0.8, MergeStrategy::Maximum);
        assert_eq!(merged, 0.8);
    }

    #[test]
    fn test_merge_local_biased() {
        let merged = merge_weight(0.4, 0.8, MergeStrategy::LocalBiased { local_weight: 0.7 });
        // 0.4 * 0.7 + 0.8 * 0.3 = 0.28 + 0.24 = 0.52
        assert!((merged - 0.52).abs() < 0.001);
    }

    #[test]
    fn test_collect_changed() {
        let current = vec![(0, 0, 0.5), (1, 0, 0.8), (2, 0, 0.3)];
        let previous = vec![(0, 0, 0.5), (1, 0, 0.5), (2, 0, 0.3)];
        let changed = collect_changed_weights(&current, &previous, 0.1);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].0, 1); // only neuron 1's weight changed enough
    }
}
