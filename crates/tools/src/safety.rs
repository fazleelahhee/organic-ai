use serde::{Deserialize, Serialize};
use organic_substrate::tile::ToolType;

/// Safety layer — limits how often external tools can be used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyLayer {
    pub action_budget: u32,
    pub actions_used: u32,
    pub read_cost: u32,
    pub write_cost: u32,
}

impl SafetyLayer {
    pub fn new(budget: u32) -> Self {
        Self {
            action_budget: budget,
            actions_used: 0,
            read_cost: 1,
            write_cost: 5,
        }
    }

    /// Check if an action is allowed and deduct from budget.
    pub fn allow_action(&mut self, tool_type: ToolType) -> bool {
        let cost = match tool_type {
            ToolType::Search | ToolType::FileSystem => self.read_cost,
            ToolType::LLM => self.write_cost,
            _ => 0, // internal tools are free
        };

        if cost == 0 {
            return true; // internal tools always allowed
        }

        if self.actions_used + cost <= self.action_budget {
            self.actions_used += cost;
            true
        } else {
            false
        }
    }

    /// Reset budget (called periodically).
    pub fn reset(&mut self) {
        self.actions_used = 0;
    }

    pub fn remaining(&self) -> u32 {
        self.action_budget.saturating_sub(self.actions_used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_tools_always_allowed() {
        let mut safety = SafetyLayer::new(0); // zero budget
        assert!(safety.allow_action(ToolType::Memory));
        assert!(safety.allow_action(ToolType::Logic));
    }

    #[test]
    fn test_external_tools_consume_budget() {
        let mut safety = SafetyLayer::new(5);
        assert!(safety.allow_action(ToolType::Search)); // costs 1
        assert_eq!(safety.remaining(), 4);
    }

    #[test]
    fn test_budget_exhaustion() {
        let mut safety = SafetyLayer::new(2);
        assert!(safety.allow_action(ToolType::Search)); // 1
        assert!(safety.allow_action(ToolType::Search)); // 2
        assert!(!safety.allow_action(ToolType::Search)); // denied
    }

    #[test]
    fn test_llm_costs_more() {
        let mut safety = SafetyLayer::new(4);
        assert!(!safety.allow_action(ToolType::LLM)); // costs 5, denied
    }

    #[test]
    fn test_reset() {
        let mut safety = SafetyLayer::new(5);
        safety.allow_action(ToolType::Search);
        safety.reset();
        assert_eq!(safety.remaining(), 5);
    }
}
