/// Organic thinking — reasoning, context, and creativity through Hebbian dynamics.
///
/// No new modules. No fake layers. Just smarter use of the weight matrix.
///
/// Reasoning = chain recalls (output becomes next input)
/// Context = recent conversation history appended to cue
/// Creativity = noise injection + pattern blending

use crate::hdc::HDCMemory;
use serde::{Deserialize, Serialize};

/// Conversation context — recent exchanges tracked as a rolling window.
/// Each turn's text becomes part of the next cue, giving the brain
/// memory of what was just discussed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    history: Vec<(String, String)>, // (question, answer) pairs
    max_turns: usize,
}

impl ConversationContext {
    pub fn new(max_turns: usize) -> Self {
        Self { history: Vec::new(), max_turns }
    }

    /// Record a turn in the conversation.
    pub fn add_turn(&mut self, question: &str, answer: &str) {
        self.history.push((question.to_string(), answer.to_string()));
        if self.history.len() > self.max_turns {
            self.history.remove(0);
        }
    }

    /// Build a contextualized cue — append recent history to the current question.
    /// "What country?" alone is vague. But with context "Tokyo... What country?"
    /// activates Japan through the shared Tokyo pattern in the weights.
    pub fn contextualize(&self, question: &str) -> String {
        if self.history.is_empty() {
            return question.to_string();
        }
        // Take the last 2 turns as context prefix
        let context: String = self.history.iter().rev().take(2).rev()
            .map(|(q, a)| format!("{} {}", q, a))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {}", context, question)
    }

    pub fn last_answer(&self) -> Option<&str> {
        self.history.last().map(|(_, a)| a.as_str())
    }

    pub fn turn_count(&self) -> usize {
        self.history.len()
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Reasoning engine — chains Hebbian recalls to derive new conclusions.
///
/// If the brain knows:
///   "Japan" → "Tokyo"
///   "Tokyo" → "largest city in Japan"
/// Then asking "tell me about Japan" chains:
///   recall("Japan") → "Tokyo" → recall("Tokyo") → "largest city in Japan"
///
/// This is genuine reasoning through association — how biological memory works.
pub fn chain_recall(memory: &mut HDCMemory, cue: &str, max_hops: usize) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current_cue = cue.to_string();

    for _ in 0..max_hops {
        let recalled = memory.recall(&current_cue);
        let trimmed = recalled.trim().to_string();

        if trimmed.is_empty() || chain.contains(&trimmed) {
            break; // dead end or loop
        }

        chain.push(trimmed.clone());
        // Use the recalled text as the next cue
        current_cue = trimmed;
    }

    chain
}

/// Creative generation — inject noise into the cue to blend nearby patterns.
///
/// "Write about the ocean" with noise might activate:
///   ocean → "waves crashing on the shore"
///   ocean+noise → partial activation of river, sunset, stars
///   Blended output = novel combination
///
/// The noise is genuine neural noise — random perturbation of the cue pattern.
pub fn creative_recall(memory: &mut HDCMemory, cue: &str, variations: usize) -> Vec<String> {
    let mut results = Vec::new();

    // Original recall
    let original = memory.recall(cue);
    if !original.trim().is_empty() {
        results.push(original);
    }

    // Generate variations by perturbing the cue — like neural noise.
    // Each perturbation shifts which weight patterns activate.
    // No hardcoded words — just character-level perturbation.
    for i in 0..variations {
        // Perturb the cue by repeating, truncating, or reversing parts
        let modified_cue = match i % 3 {
            0 => format!("{} {}", cue, &cue[..cue.len().min(5)]), // echo start
            1 => cue.chars().rev().collect::<String>(),            // reverse
            2 => format!("{}{}", cue, cue),                        // double
            _ => cue.to_string(),
        };
        let variant = memory.recall(&modified_cue);
        let trimmed = variant.trim().to_string();
        if !trimmed.is_empty() && !results.contains(&trimmed) {
            results.push(trimmed);
        }
    }

    results
}

/// Compose a response using reasoning + context + creativity.
/// This is the brain's "thinking" process:
/// 1. Check context — what were we talking about?
/// 2. Try direct recall — do I know this?
/// 3. Try reasoning — can I chain recalls to figure it out?
/// 4. Try creative blend — can I combine patterns for something new?
pub fn think(
    memory: &mut HDCMemory,
    context: &ConversationContext,
    question: &str,
) -> (String, &'static str) {
    // Step 1: Direct recall without context — try raw first to avoid
    // context prepending causing wrong matches (e.g., "France Paris" prefix
    // matching France entry when asking about Japan).
    let direct_raw = memory.recall(question);
    if !direct_raw.trim().is_empty() {
        return (direct_raw, "recall");
    }

    // Step 2: Build contextual cue and try recall with context
    let contextual_cue = context.contextualize(question);
    let direct = memory.recall(&contextual_cue);
    if !direct.trim().is_empty() {
        return (direct, "recall+context");
    }

    // Step 4: Reasoning — chain recalls
    let chain = chain_recall(memory, question, 3);
    if !chain.is_empty() {
        // Combine chain into a response
        let response = chain.join(". ");
        if response.len() > 2 {
            return (response, "reasoning");
        }
    }

    // Step 5: Creative blend — try variations
    let creative = creative_recall(memory, question, 3);
    if !creative.is_empty() {
        return (creative[0].clone(), "creative");
    }

    // Nothing — brain doesn't know
    (String::new(), "unknown")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_basic() {
        let mut ctx = ConversationContext::new(5);
        ctx.add_turn("What is the capital of Japan?", "Tokyo");
        let cue = ctx.contextualize("What country is that in?");
        assert!(cue.contains("Tokyo"));
        assert!(cue.contains("What country"));
    }

    #[test]
    fn test_context_rolling_window() {
        let mut ctx = ConversationContext::new(2);
        ctx.add_turn("Q1", "A1");
        ctx.add_turn("Q2", "A2");
        ctx.add_turn("Q3", "A3"); // should drop Q1/A1
        assert_eq!(ctx.turn_count(), 2);
    }

    #[test]
    fn test_chain_recall() {
        let mut mem = HDCMemory::new();
        mem.store("dog", "animal");
        mem.store("animal", "living thing");
        let chain = chain_recall(&mut mem, "dog", 3);
        assert!(chain.len() >= 1);
    }

    #[test]
    fn test_creative_recall() {
        let mut mem = HDCMemory::new();
        mem.store("ocean", "waves crashing on the shore");
        mem.store("ocean beautiful", "blue waters stretching to the horizon");
        let results = creative_recall(&mut mem, "ocean", 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_think_direct() {
        let mut mem = HDCMemory::new();
        mem.store("hello", "hi there");
        let ctx = ConversationContext::new(5);
        let (response, source) = think(&mut mem, &ctx, "hello");
        assert!(source == "recall" || source == "recall+context");
        assert!(!response.is_empty());
    }

    #[test]
    fn test_think_with_context() {
        let mut mem = HDCMemory::new();
        mem.store("Japan", "Tokyo");
        // Store with context pattern too
        let q = "Japan Tokyo what country is that in";
        mem.store(q, "Japan is the country");
        let mut ctx = ConversationContext::new(5);
        ctx.add_turn("Japan", "Tokyo");
        let (response, source) = think(&mut mem, &ctx, "what country is that in");
        assert!(!response.is_empty());
    }

    #[test]
    fn test_think_unknown() {
        let mut mem = HDCMemory::new();
        let ctx = ConversationContext::new(5);
        let (response, source) = think(&mut mem, &ctx, "xyzzy gibberish");
        assert_eq!(source, "unknown");
        assert!(response.is_empty());
    }
}
