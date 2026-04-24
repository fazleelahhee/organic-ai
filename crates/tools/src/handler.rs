use organic_substrate::tile::ToolType;
use crate::memory::ToolMemory;
use crate::pattern::pattern_similarity;
use crate::logic::{apply_logic, LogicOp};
use crate::external;
use crate::safety::SafetyLayer;

/// Global tool state — shared across all organisms.
pub struct ToolHandler {
    pub memory: ToolMemory,
    pub safety: SafetyLayer,
}

impl ToolHandler {
    pub fn new() -> Self {
        Self {
            memory: ToolMemory::new(100),
            safety: SafetyLayer::new(50), // 50 external actions per reset cycle
        }
    }

    /// Process a tool interaction. Takes the tool type and input signals,
    /// returns an output signal value.
    pub fn interact(&mut self, tool_type: ToolType, input: &[f32]) -> f32 {
        if !self.safety.allow_action(tool_type) {
            return 0.0; // denied by safety layer
        }
        match tool_type {
            ToolType::Memory => {
                if input.len() >= 2 {
                    // input[0] = address, input[1] = store/retrieve flag
                    if input[1] > 0.5 && input.len() >= 3 {
                        // Store mode
                        self.memory.store(input[0], input[2..].to_vec());
                        1.0 // success signal
                    } else {
                        // Retrieve mode
                        let pattern = self.memory.retrieve(input[0]);
                        if pattern.is_empty() { 0.0 } else { pattern[0] }
                    }
                } else {
                    0.0
                }
            }
            ToolType::Pattern => {
                if input.len() >= 4 {
                    let mid = input.len() / 2;
                    pattern_similarity(&input[..mid], &input[mid..])
                } else {
                    0.0
                }
            }
            ToolType::Logic => {
                if input.len() >= 2 {
                    // Use first input to select operation
                    let op_idx = (input[0] * 5.0) as usize % 5;
                    let op = match op_idx {
                        0 => LogicOp::And,
                        1 => LogicOp::Or,
                        2 => LogicOp::Xor,
                        3 => LogicOp::Add,
                        _ => LogicOp::Multiply,
                    };
                    let a = if input.len() > 1 { input[1] } else { 0.0 };
                    let b = if input.len() > 2 { input[2] } else { 0.0 };
                    apply_logic(op, a, b)
                } else {
                    0.0
                }
            }
            ToolType::Language => {
                // Proto-language: convert signal pattern to a hash-like token
                if input.is_empty() { return 0.0; }
                let sum: f32 = input.iter().sum();
                (sum / input.len() as f32).clamp(0.0, 1.0)
            }
            ToolType::Search => {
                let query = external::signals_to_query(input);
                let result = external::web_search(&query);
                result.signal_value
            }
            ToolType::LLM => {
                let prompt = external::signals_to_query(input);
                let result = external::llm_query(&prompt);
                result.signal_value
            }
            ToolType::FileSystem => {
                let filename = external::signals_to_query(input);
                let result = external::read_file(&filename);
                result.signal_value
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_store_retrieve() {
        let mut handler = ToolHandler::new();
        // Store
        let result = handler.interact(ToolType::Memory, &[0.5, 0.8, 1.0, 2.0]);
        assert_eq!(result, 1.0);
        // Retrieve
        let result = handler.interact(ToolType::Memory, &[0.5, 0.2]);
        assert_eq!(result, 1.0); // first element of stored pattern
    }

    #[test]
    fn test_pattern_comparison() {
        let mut handler = ToolHandler::new();
        let result = handler.interact(ToolType::Pattern, &[1.0, 0.0, 1.0, 0.0]);
        assert!((result - 1.0).abs() < 0.01); // identical halves
    }

    #[test]
    fn test_logic_operation() {
        let mut handler = ToolHandler::new();
        // op_idx = (0.0 * 5.0) as usize % 5 = 0 = And
        let result = handler.interact(ToolType::Logic, &[0.0, 0.8, 0.9]);
        assert_eq!(result, 1.0); // And(0.8, 0.9) = 1.0
    }
}
