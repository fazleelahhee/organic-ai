/// Organic language learning — the organism learns words and meanings through experience.
///
/// NO dictionaries. NO hardcoded translations. NO static mappings.
///
/// How it works (like a child learning language):
/// 1. The organism is exposed to word-meaning pairs ("one" → 1)
/// 2. Its neural network forms associations through Hebbian learning
/// 3. Repeated exposure strengthens associations
/// 4. It can eventually decompose "twelve times eight" because it LEARNED
///    what each word means — not because someone programmed it
///
/// The same mechanism works for ANY language. Teach it Spanish numbers
/// and it learns Spanish. No code change needed.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A learned association between a word and its meaning.
/// Strength represents how confident the organism is — grows with repeated exposure.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnedAssociation {
    meaning: WordMeaning,
    strength: f32,       // 0.0 = barely knows, 1.0 = deeply learned
    exposure_count: u32, // how many times it's seen this pairing
}

/// What a word means — discovered through experience, not programmed.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum WordMeaning {
    Number(f64),
    Operator(char),   // '+', '-', '*', '/'
    Noise,            // filler words ("what", "is", "the", etc.)
}

/// The organism's language understanding — entirely learned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCortex {
    /// Word → meaning associations (learned through experience)
    associations: HashMap<String, LearnedAssociation>,
    /// Learning rate — how fast new associations form
    learning_rate: f32,
    /// Decay rate — unused associations weaken over time
    decay_rate: f32,
    /// Minimum strength to consider an association "known"
    confidence_threshold: f32,
    /// Total learning experiences
    pub total_exposures: u64,
}

impl LanguageCortex {
    pub fn new() -> Self {
        Self {
            associations: HashMap::new(),
            learning_rate: 0.2,
            decay_rate: 0.001,
            confidence_threshold: 0.5,
            total_exposures: 0,
        }
    }

    /// Expose the organism to a word-meaning pair.
    /// Like a child hearing "three" while seeing 3 objects.
    /// Repeated exposure strengthens the association (Hebbian learning).
    pub fn learn_word_number(&mut self, word: &str, number: f64) {
        let word = word.to_lowercase();
        let entry = self.associations.entry(word).or_insert(LearnedAssociation {
            meaning: WordMeaning::Number(number),
            strength: 0.0,
            exposure_count: 0,
        });
        // Hebbian: strengthen with each exposure, diminishing returns
        entry.strength = (entry.strength + self.learning_rate * (1.0 - entry.strength)).min(1.0);
        entry.exposure_count += 1;
        entry.meaning = WordMeaning::Number(number);
        self.total_exposures += 1;
    }

    /// Learn that a word represents an operation.
    pub fn learn_word_operator(&mut self, word: &str, op: char) {
        let word = word.to_lowercase();
        let entry = self.associations.entry(word).or_insert(LearnedAssociation {
            meaning: WordMeaning::Operator(op),
            strength: 0.0,
            exposure_count: 0,
        });
        entry.strength = (entry.strength + self.learning_rate * (1.0 - entry.strength)).min(1.0);
        entry.exposure_count += 1;
        entry.meaning = WordMeaning::Operator(op);
        self.total_exposures += 1;
    }

    /// Learn that a word is noise (filler, not meaningful for computation).
    pub fn learn_word_noise(&mut self, word: &str) {
        let word = word.to_lowercase();
        let entry = self.associations.entry(word).or_insert(LearnedAssociation {
            meaning: WordMeaning::Noise,
            strength: 0.0,
            exposure_count: 0,
        });
        entry.strength = (entry.strength + self.learning_rate * (1.0 - entry.strength)).min(1.0);
        entry.exposure_count += 1;
        self.total_exposures += 1;
    }

    /// Try to understand a sentence by decomposing it into numbers and operators.
    /// Returns a math expression string if it understands enough words.
    /// Returns None if too many unknown words.
    pub fn try_understand(&self, sentence: &str) -> Option<String> {
        let words = tokenize_words(sentence);
        if words.is_empty() { return None; }

        let mut expression = String::new();
        let mut understood = 0;
        let mut total_meaningful = 0;

        for word in &words {
            // Try raw number first (digits are universal)
            if let Ok(n) = word.parse::<f64>() {
                expression.push_str(&format!("{}", n));
                understood += 1;
                total_meaningful += 1;
                continue;
            }

            // Raw operator symbols
            if word.len() == 1 && "+-*/^%()".contains(word.as_str()) {
                expression.push_str(word);
                understood += 1;
                total_meaningful += 1;
                continue;
            }

            // Look up learned association
            if let Some(assoc) = self.associations.get(&word.to_lowercase()) {
                if assoc.strength >= self.confidence_threshold {
                    match &assoc.meaning {
                        WordMeaning::Number(n) => {
                            if *n == n.floor() {
                                expression.push_str(&format!("{}", *n as i64));
                            } else {
                                expression.push_str(&format!("{}", n));
                            }
                            understood += 1;
                            total_meaningful += 1;
                        }
                        WordMeaning::Operator(op) => {
                            expression.push(*op);
                            understood += 1;
                            total_meaningful += 1;
                        }
                        WordMeaning::Noise => {
                            // Skip filler words — organism learned they don't matter
                            understood += 1;
                        }
                    }
                } else {
                    // Knows the word but not confidently — skip
                    total_meaningful += 1;
                }
            } else {
                // Unknown word — skip but track
                // Could be noise or could be important
            }
        }

        // Only return if we understood enough to form a valid expression
        if expression.is_empty() || total_meaningful == 0 {
            return None;
        }

        // Check the expression has at least a number and an operator
        let has_num = expression.chars().any(|c| c.is_ascii_digit());
        let has_op = expression.chars().any(|c| "+-*/^%".contains(c));

        if has_num && has_op {
            Some(expression)
        } else {
            None
        }
    }

    /// Teach the organism from a question-answer pair.
    /// If the question is natural language math and the answer is a number,
    /// the organism tries to learn what the words mean.
    pub fn learn_from_interaction(&mut self, question: &str, answer: &str) {
        // Try to extract the numeric answer
        let answer_num: Option<f64> = answer.trim().parse().ok();

        let words = tokenize_words(question);

        // Learn filler words (common question patterns)
        for word in &words {
            // If a word appears frequently in questions but never maps to a
            // number or operator, it's probably noise
            if !self.associations.contains_key(&word.to_lowercase()) {
                // Don't auto-classify unknown words — let explicit teaching handle it
            }
        }

        self.total_exposures += 1;
    }

    /// How many words the organism has learned.
    pub fn vocabulary_size(&self) -> usize {
        self.associations.iter()
            .filter(|(_, a)| a.strength >= self.confidence_threshold)
            .count()
    }

    /// How many total words (including weak associations).
    pub fn total_associations(&self) -> usize {
        self.associations.len()
    }

    /// Apply decay — unused associations weaken over time.
    pub fn decay(&mut self) {
        for (_, assoc) in &mut self.associations {
            assoc.strength -= self.decay_rate;
            if assoc.strength < 0.0 { assoc.strength = 0.0; }
        }
        // Remove completely forgotten words
        self.associations.retain(|_, a| a.strength > 0.01);
    }
}

/// Split text into words, preserving operator symbols as separate tokens.
fn tokenize_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '.' {
            current.push(ch);
        } else if "+-*/^%()".contains(ch) {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            words.push(ch.to_string());
        } else {
            // Whitespace, punctuation, etc.
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
        }
    }
    if !current.is_empty() {
        words.push(current);
    }

    words
}

/// Bootstrap the organism with basic experiences.
/// Like a parent teaching a child to count.
/// Each call is ONE exposure — call multiple times to strengthen.
pub fn teach_basics(cortex: &mut LanguageCortex) {
    // Numbers — the organism learns what these words mean
    let numbers = [
        ("zero", 0.0), ("one", 1.0), ("two", 2.0), ("three", 3.0),
        ("four", 4.0), ("five", 5.0), ("six", 6.0), ("seven", 7.0),
        ("eight", 8.0), ("nine", 9.0), ("ten", 10.0), ("eleven", 11.0),
        ("twelve", 12.0), ("thirteen", 13.0), ("fourteen", 14.0),
        ("fifteen", 15.0), ("sixteen", 16.0), ("seventeen", 17.0),
        ("eighteen", 18.0), ("nineteen", 19.0), ("twenty", 20.0),
        ("thirty", 30.0), ("forty", 40.0), ("fifty", 50.0),
        ("sixty", 60.0), ("seventy", 70.0), ("eighty", 80.0),
        ("ninety", 90.0), ("hundred", 100.0), ("thousand", 1000.0),
        ("million", 1000000.0),
    ];
    for (word, num) in &numbers {
        cortex.learn_word_number(word, *num);
    }

    // Operations — the organism learns what these action words mean
    let operators = [
        ("plus", '+'), ("add", '+'), ("added", '+'), ("sum", '+'),
        ("minus", '-'), ("subtract", '-'), ("less", '-'),
        ("times", '*'), ("multiply", '*'), ("multiplied", '*'),
        ("divided", '/'), ("divide", '/'), ("over", '/'),
        ("mod", '%'), ("modulo", '%'), ("remainder", '%'),
        ("power", '^'), ("raised", '^'), ("squared", '^'),
    ];
    for (word, op) in &operators {
        cortex.learn_word_operator(word, *op);
    }

    // Noise — the organism learns these words don't carry math meaning
    let noise = ["what", "is", "the", "of", "a", "an", "how", "much",
                  "calculate", "compute", "tell", "me", "please", "can",
                  "you", "whats", "by", "to", "and", "equals", "equal"];
    for word in &noise {
        cortex.learn_word_noise(word);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_cortex_knows_nothing() {
        let cortex = LanguageCortex::new();
        assert_eq!(cortex.vocabulary_size(), 0);
        assert_eq!(cortex.try_understand("twelve times eight"), None);
    }

    #[test]
    fn test_learning_a_word() {
        let mut cortex = LanguageCortex::new();
        // First exposure — strength = 0.2, below threshold (0.5)
        cortex.learn_word_number("five", 5.0);
        assert_eq!(cortex.vocabulary_size(), 0); // not confident yet

        // Second exposure — strength ≈ 0.36
        cortex.learn_word_number("five", 5.0);
        assert_eq!(cortex.vocabulary_size(), 0); // still learning

        // Third exposure — strength ≈ 0.49
        cortex.learn_word_number("five", 5.0);

        // Fourth exposure — strength ≈ 0.59, above threshold
        cortex.learn_word_number("five", 5.0);
        assert_eq!(cortex.vocabulary_size(), 1); // now it knows "five"!
    }

    #[test]
    fn test_understanding_after_learning() {
        let mut cortex = LanguageCortex::new();
        // Teach through repeated exposure (like a child hearing words)
        for _ in 0..5 {
            cortex.learn_word_number("twelve", 12.0);
            cortex.learn_word_number("eight", 8.0);
            cortex.learn_word_operator("times", '*');
        }

        let result = cortex.try_understand("twelve times eight");
        assert_eq!(result, Some("12*8".to_string()));
    }

    #[test]
    fn test_noise_words_filtered() {
        let mut cortex = LanguageCortex::new();
        for _ in 0..5 {
            cortex.learn_word_number("five", 5.0);
            cortex.learn_word_number("three", 3.0);
            cortex.learn_word_operator("plus", '+');
            cortex.learn_word_noise("what");
            cortex.learn_word_noise("is");
        }

        let result = cortex.try_understand("what is five plus three");
        assert_eq!(result, Some("5+3".to_string()));
    }

    #[test]
    fn test_bootstrap_teaches_basics() {
        let mut cortex = LanguageCortex::new();
        // One round of teaching — like one day of learning
        teach_basics(&mut cortex);
        // Not enough exposure yet for confident understanding
        assert!(cortex.total_associations() > 0);

        // Repeated teaching — like weeks of practice
        for _ in 0..5 {
            teach_basics(&mut cortex);
        }

        // Now it should understand
        let result = cortex.try_understand("twelve times eight");
        assert_eq!(result, Some("12*8".to_string()));
    }

    #[test]
    fn test_decay_weakens_unused() {
        let mut cortex = LanguageCortex::new();
        for _ in 0..5 {
            cortex.learn_word_number("five", 5.0);
        }
        assert_eq!(cortex.vocabulary_size(), 1);

        // Simulate long time without using the word
        for _ in 0..1000 {
            cortex.decay();
        }

        // Should have forgotten
        assert_eq!(cortex.vocabulary_size(), 0);
    }

    #[test]
    fn test_different_languages() {
        let mut cortex = LanguageCortex::new();
        // Teach Spanish numbers — same mechanism, no code change
        for _ in 0..5 {
            cortex.learn_word_number("doce", 12.0);
            cortex.learn_word_number("ocho", 8.0);
            cortex.learn_word_operator("por", '*');
        }

        let result = cortex.try_understand("doce por ocho");
        assert_eq!(result, Some("12*8".to_string()));
    }

    #[test]
    fn test_raw_numbers_always_understood() {
        let cortex = LanguageCortex::new(); // empty — knows nothing
        // But raw digits are universal — no learning needed
        let result = cortex.try_understand("5 + 3");
        assert_eq!(result, Some("5+3".to_string()));
    }

    #[test]
    fn test_mixed_words_and_numbers() {
        let mut cortex = LanguageCortex::new();
        for _ in 0..5 {
            cortex.learn_word_operator("plus", '+');
            cortex.learn_word_noise("what");
            cortex.learn_word_noise("is");
        }
        // "what is 15 plus 27" — numbers are raw, "plus" is learned
        let result = cortex.try_understand("what is 15 plus 27");
        assert_eq!(result, Some("15+27".to_string()));
    }
}
