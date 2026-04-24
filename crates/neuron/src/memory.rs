/// Hopfield Attractor Memory — genuine organic memory through synaptic weights.
///
/// NO Vec of stored pairs. NO cosine similarity search. NO string storage.
///
/// How it works (like hippocampal memory):
/// 1. Text is encoded as a SPARSE BINARY PATTERN across neurons
/// 2. Associations are stored by modifying SYNAPTIC WEIGHTS (Hebbian rule)
/// 3. Recall = present cue → multiply through weight matrix → decode
/// 4. The weight matrix IS the memory
///
/// Architecture:
/// - CUE layer: sparse distributed code (hash-based, like cortical input)
/// - MEMORY layer: topographic code (organized by position and character,
///   like sensory cortex)
/// - HETERO-ASSOCIATIVE weight matrix W: CUE→MEMORY (Hebbian learned)
///
/// Memory layer organization (topographic):
///   neuron_index = position * NUM_CHARS + (byte - 32) * NEURONS_PER_CHAR_SLOT + k
///   Each (position, character) pair owns NEURONS_PER_CHAR_SLOT dedicated neurons.
///   NO hash collisions. NO overlap between characters.
///
/// Cue layer: sparse distributed hash-based code (like cortical input).
///   Each (byte, position) pair activates NEURONS_PER_CUE_CHAR neurons via hash.
///
/// Storage: W += lr * outer(cue_pattern, mem_pattern)
/// Recall:  drive = W^T * cue_pattern → decode from topographic map

use serde::{Deserialize, Serialize};

/// Number of printable ASCII characters (32..127)
const NUM_CHARS: usize = 95;

/// Maximum text length.
const MAX_TEXT_LEN: usize = 128;

/// How many dedicated neurons per (position, character) in the memory layer.
/// Memory layer size = MAX_TEXT_LEN * NUM_CHARS * NEURONS_PER_SLOT
/// = 128 * 95 * 2 = 24320 neurons
const NEURONS_PER_SLOT: usize = 2;

/// Memory layer size (derived).
const MEM_SIZE: usize = MAX_TEXT_LEN * NUM_CHARS * NEURONS_PER_SLOT; // 24320

/// Cue layer size — sparse distributed code.
const CUE_SIZE: usize = 16384;

/// Neurons activated per (byte, position) in the cue layer.
/// Lower = sparser = less cross-talk between stored patterns.
const NEURONS_PER_CUE_CHAR: usize = 6;

/// Topographic index for memory encoding: NO hashing, NO collisions.
/// Each (position, character, slot) maps to a unique neuron.
fn mem_index(byte: u8, pos: usize, slot: usize) -> usize {
    let char_idx = (byte.saturating_sub(32)) as usize;
    pos * NUM_CHARS * NEURONS_PER_SLOT + char_idx * NEURONS_PER_SLOT + slot
}

/// Encode cue text as sparse distributed pattern using TRIGRAM hashing.
///
/// Instead of hashing individual characters (which causes "Japan" and "Spain"
/// to share neurons for 'a' at position 2), we hash TRIGRAMS: each character
/// combined with its neighbors. This makes the representation context-sensitive,
/// like how real cortical neurons respond to feature combinations, not
/// individual features.
///
/// A word of length L produces L trigrams, each activating NEURONS_PER_CUE_CHAR
/// neurons. Trigrams from different words almost never collide.
fn cue_pattern(text: &str) -> Vec<f32> {
    let mut pattern = vec![0.0f32; CUE_SIZE];
    let bytes: Vec<u8> = text.bytes().collect();
    let len = bytes.len().min(MAX_TEXT_LEN);

    for pos in 0..len {
        // Trigram: (prev_char, current_char, next_char)
        // At boundaries, use 0 as sentinel.
        let prev = if pos > 0 { bytes[pos - 1] } else { 0 };
        let curr = bytes[pos];
        let next = if pos + 1 < len { bytes[pos + 1] } else { 0 };

        for k in 0..NEURONS_PER_CUE_CHAR {
            // Hash the trigram + position + slot
            let h = (prev as u64)
                .wrapping_mul(2654435761)
                .wrapping_add((curr as u64).wrapping_mul(1640531527))
                .wrapping_add((next as u64).wrapping_mul(2246822519))
                .wrapping_add((pos as u64).wrapping_mul(40503))
                .wrapping_add((k as u64).wrapping_mul(104729))
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = ((h >> 17) as usize) % CUE_SIZE;
            pattern[idx] = 1.0;
        }
    }
    pattern
}

/// Encode memory text as topographic pattern (no collisions).
fn mem_pattern(text: &str) -> Vec<f32> {
    let mut pattern = vec![0.0f32; MEM_SIZE];
    for (pos, byte) in text.bytes().enumerate() {
        if pos >= MAX_TEXT_LEN { break; }
        if byte < 32 || byte >= 127 { continue; }
        for k in 0..NEURONS_PER_SLOT {
            let idx = mem_index(byte, pos, k);
            if idx < MEM_SIZE {
                pattern[idx] = 1.0;
            }
        }
    }
    pattern
}

/// Hopfield Attractor Memory — stores associations as synaptic weights.
///
/// The ONLY data structure is a weight matrix. No stored strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttractorMemory {
    /// Hetero-associative weight matrix: w[cue_i * MEM_SIZE + mem_j].
    /// This IS the memory. Associations exist only as weight values.
    w: Vec<f32>,

    /// Learning rate.
    learning_rate: f32,

    /// Global weight decay per storage event (palimpsest property).
    decay_rate: f32,

    /// How many patterns have been stored.
    pub patterns_stored: u64,
}

impl AttractorMemory {
    pub fn new() -> Self {
        Self {
            w: vec![0.0f32; CUE_SIZE * MEM_SIZE],
            learning_rate: 1.0,
            decay_rate: 0.002,
            patterns_stored: 0,
        }
    }

    /// Store an association: cue_text → memory_text.
    ///
    /// Modifies synaptic weights using the COVARIANCE LEARNING RULE:
    ///   W[i][j] += lr * cue[i] * (mem[j] - mean_mem)
    ///
    /// This is biologically plausible (BCM theory) and mathematically
    /// superior to plain Hebbian: it STRENGTHENS weights to target neurons
    /// AND WEAKENS weights to non-target neurons at each position.
    ///
    /// Result: when cue is presented, target memory neurons get strong
    /// positive drive while non-target neurons get suppressed. This
    /// eliminates cross-talk at unstored positions.
    pub fn store(&mut self, cue_text: &str, memory_text: &str) {
        let cue = cue_pattern(cue_text);
        let mem = mem_pattern(memory_text);

        // Global decay — palimpsest property
        let factor = 1.0 - self.decay_rate;
        for w in &mut self.w {
            *w *= factor;
        }

        // Compute mean activity in the memory pattern.
        // For sparse patterns this is small (e.g. 40 active / 12160 total ≈ 0.003).
        let mem_active: f32 = mem.iter().sum();
        let mem_mean = mem_active / MEM_SIZE as f32;

        // Covariance rule: W += lr * cue * (mem - mean)
        // For active memory neurons: delta = lr * 1.0 * (1.0 - mean) ≈ +lr
        // For inactive memory neurons: delta = lr * 1.0 * (0.0 - mean) ≈ -lr*mean (small negative)
        // This creates INHIBITORY weights that suppress non-target chars.
        for i in 0..CUE_SIZE {
            if cue[i] < 0.5 { continue; }
            let row_start = i * MEM_SIZE;
            for j in 0..MEM_SIZE {
                let delta = self.learning_rate * cue[i] * (mem[j] - mem_mean);
                self.w[row_start + j] += delta;
            }
        }

        self.patterns_stored += 1;
    }

    /// Recall: present cue, compute drive through weights, decode.
    pub fn recall(&self, cue_text: &str) -> String {
        let cue = cue_pattern(cue_text);

        // Matrix-vector multiply: drive[j] = sum_i(W[i][j] * cue[i])
        let mut drive = vec![0.0f32; MEM_SIZE];
        for i in 0..CUE_SIZE {
            if cue[i] < 0.5 { continue; }
            let row_start = i * MEM_SIZE;
            for j in 0..MEM_SIZE {
                drive[j] += self.w[row_start + j];
            }
        }

        // Decode from topographic map.
        // For each position, compute the score for every character,
        // subtract the MEAN score (removes cross-talk DC component),
        // then pick the winner if it has sufficient margin.
        let mut result = String::new();

        for pos in 0..MAX_TEXT_LEN {
            // First pass: compute raw scores for all characters at this position
            let mut scores = [0.0f32; NUM_CHARS];
            for char_offset in 0..NUM_CHARS {
                let mut score = 0.0f32;
                for k in 0..NEURONS_PER_SLOT {
                    let idx = pos * NUM_CHARS * NEURONS_PER_SLOT
                        + char_offset * NEURONS_PER_SLOT + k;
                    if idx < MEM_SIZE {
                        score += drive[idx];
                    }
                }
                scores[char_offset] = score;
            }

            // Compute mean score across all characters at this position.
            // Cross-talk from other stored patterns raises ALL characters
            // approximately equally. Subtracting the mean removes this bias.
            let mean: f32 = scores.iter().sum::<f32>() / NUM_CHARS as f32;

            // Subtract mean and find winner
            let mut best_char: u8 = 0;
            let mut best_score: f32 = f32::NEG_INFINITY;
            let mut second_score: f32 = f32::NEG_INFINITY;

            for char_offset in 0..NUM_CHARS {
                let centered = scores[char_offset] - mean;
                if centered > best_score {
                    second_score = best_score;
                    best_score = centered;
                    best_char = (char_offset + 32) as u8;
                } else if centered > second_score {
                    second_score = centered;
                }
            }

            // After mean subtraction:
            // - At stored positions: the correct char is far above mean (high signal).
            //   Its centered score is large positive. Runner-up is near zero.
            // - At unstored positions: all chars are near the mean.
            //   Centered scores are all small. Best and second-best are close.
            let margin = best_score - second_score;
            if best_score > 0.5 && margin > 0.3 && best_char > b' ' {
                result.push(best_char as char);
            }
        }

        result.trim_end().to_string()
    }

    /// Strengthen a memory by re-storing it (like biological rehearsal).
    pub fn rehearse(&mut self, cue_text: &str, memory_text: &str, repetitions: usize) {
        for _ in 0..repetitions {
            self.store(cue_text, memory_text);
        }
    }

    /// Memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let nonzero = self.w.iter().filter(|&&w| w.abs() > 0.001).count();
        let avg: f32 = if nonzero > 0 {
            self.w.iter().filter(|&&w| w.abs() > 0.001).sum::<f32>() / nonzero as f32
        } else { 0.0 };

        MemoryStats {
            cue_size: CUE_SIZE,
            mem_size: MEM_SIZE,
            patterns_stored: self.patterns_stored,
            nonzero_weights: nonzero,
            avg_weight: avg,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub cue_size: usize,
    pub mem_size: usize,
    pub patterns_stored: u64,
    pub nonzero_weights: usize,
    pub avg_weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_memory() -> AttractorMemory {
        AttractorMemory::new()
    }

    #[test]
    fn test_creation() {
        let mem = test_memory();
        assert_eq!(mem.patterns_stored, 0);
    }

    #[test]
    fn test_cue_encoding_deterministic() {
        let p1 = cue_pattern("Tokyo");
        let p2 = cue_pattern("Tokyo");
        assert_eq!(p1, p2, "Same text must produce same pattern");
    }

    #[test]
    fn test_cue_encoding_differs() {
        let p1 = cue_pattern("Tokyo");
        let p2 = cue_pattern("Paris");
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_cue_encoding_position_sensitive() {
        let p1 = cue_pattern("ab");
        let p2 = cue_pattern("ba");
        assert_ne!(p1, p2, "Position matters");
    }

    #[test]
    fn test_mem_encoding_no_collisions() {
        // Every (position, character) pair should map to unique neurons
        // in the topographic memory layer.
        let p1 = mem_pattern("a");  // 'a' at position 0
        let p2 = mem_pattern("b");  // 'b' at position 0
        // They should have zero overlap
        let overlap: f32 = p1.iter().zip(p2.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(overlap, 0.0, "Different chars at same position must not overlap");
    }

    #[test]
    fn test_encoding_sparsity() {
        let p = cue_pattern("Tokyo");
        let active = p.iter().filter(|&&v| v > 0.5).count();
        let sparsity = active as f32 / CUE_SIZE as f32;
        assert!(sparsity < 0.1, "Pattern should be sparse, got {:.3}", sparsity);
        assert!(active > 10, "Pattern should have some active neurons, got {}", active);
    }

    #[test]
    fn test_store_modifies_weights() {
        let mut mem = test_memory();
        let w_before: f32 = mem.w.iter().sum();
        mem.store("Japan", "Tokyo");
        let w_after: f32 = mem.w.iter().sum();
        assert_ne!(w_before, w_after, "Storing must modify weights");
    }

    #[test]
    fn test_recall_single_fact() {
        let mut mem = test_memory();
        mem.rehearse("Japan", "Tokyo", 5);
        let result = mem.recall("Japan");
        assert_eq!(result, "Tokyo",
            "Single fact recall failed: got '{}' instead of 'Tokyo'", result);
    }

    #[test]
    fn test_recall_multiple_facts() {
        let mut mem = test_memory();
        let facts = vec![
            ("Japan", "Tokyo"),
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
        ];
        for &(cue, val) in &facts {
            mem.rehearse(cue, val, 5);
        }
        let mut correct = 0;
        for &(cue, expected) in &facts {
            let result = mem.recall(cue);
            if result == expected { correct += 1; }
            else {
                eprintln!("MISS: '{}' -> '{}' (expected '{}')", cue, result, expected);
            }
        }
        assert!(correct >= 4,
            "Should recall at least 4/5 facts correctly, got {}/5", correct);
    }

    #[test]
    fn test_recall_20_facts() {
        let mut mem = test_memory();
        let facts = vec![
            ("Japan", "Tokyo"),
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("UK", "London"),
            ("Russia", "Moscow"),
            ("China", "Beijing"),
            ("India", "Delhi"),
            ("Brazil", "Brasilia"),
            ("Egypt", "Cairo"),
            ("Mexico", "MexicoCity"),
            ("Canada", "Ottawa"),
            ("Turkey", "Ankara"),
            ("Kenya", "Nairobi"),
            ("Peru", "Lima"),
            ("Cuba", "Havana"),
            ("Nepal", "Kathmandu"),
            ("Iraq", "Baghdad"),
            ("Chile", "Santiago"),
        ];
        for &(cue, val) in &facts {
            mem.rehearse(cue, val, 5);
        }
        let mut correct = 0;
        for &(cue, expected) in &facts {
            let result = mem.recall(cue);
            if result == expected { correct += 1; }
            else {
                eprintln!("MISS: '{}' -> '{}' (expected '{}')", cue, result, expected);
            }
        }
        assert!(correct >= 15,
            "Should recall at least 15/20 facts, got {}/20", correct);
    }

    #[test]
    fn test_no_string_storage() {
        let mut mem = test_memory();
        mem.store("Japan", "Tokyo");
        // w is Vec<f32> — pure weights, no strings
        assert_eq!(mem.w.len(), CUE_SIZE * MEM_SIZE);
    }

    #[test]
    fn test_different_cues_different_results() {
        let mut mem = test_memory();
        mem.rehearse("Japan", "Tokyo", 5);
        mem.rehearse("France", "Paris", 5);
        let r1 = mem.recall("Japan");
        let r2 = mem.recall("France");
        assert_ne!(r1, r2, "Different cues must produce different recalls");
    }

    #[test]
    fn test_stats() {
        let mut mem = test_memory();
        mem.store("Japan", "Tokyo");
        let stats = mem.stats();
        assert_eq!(stats.patterns_stored, 1);
        assert!(stats.nonzero_weights > 0);
    }
}
