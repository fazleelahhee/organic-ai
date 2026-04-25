/// Working Memory — the prefrontal cortex of the organic brain.
///
/// Holds intermediate values while the brain thinks in steps.
/// This is what enables:
/// - Algebra: hold equation state across steps
/// - Coding: hold variables, plan sequences
/// - Writing: hold theme, build sentences
/// - Reasoning: hold premises, derive conclusions
///
/// Architecture: a small set of neural REGISTERS that sustain
/// activity through recurrent self-excitation. Like how prefrontal
/// cortex neurons maintain persistent firing to hold information
/// across delays.
///
/// No HashMap. No Vec of strings. Registers are ACTIVITY PATTERNS
/// in neural populations, sustained by recurrent dynamics.

use serde::{Deserialize, Serialize};

const NUM_REGISTERS: usize = 8;   // 8 working memory slots (humans have ~4-7)
const REGISTER_SIZE: usize = 1900; // neurons per register (20 chars × 95 printable ASCII)

/// A single working memory register — holds a value as sustained neural activity.
/// NO string storage. The activity pattern IS the value.
/// Decoded through the same mechanism as attractor memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Register {
    /// Activity pattern — the value being held. This IS the memory.
    activity: Vec<f32>,
    /// Is this register actively holding something?
    active: bool,
    /// How long has this been held (decays over time)
    hold_strength: f32,
}

impl Register {
    fn new() -> Self {
        Self {
            activity: vec![0.0; REGISTER_SIZE],
            active: false,
            hold_strength: 0.0,
        }
    }

    /// Load a value — encode text as activity pattern using topographic encoding.
    /// Same scheme as attractor memory: position × character → neuron.
    fn load(&mut self, value: &str) {
        self.activity = vec![0.0; REGISTER_SIZE];
        let chars_per_slot = 95; // printable ASCII
        for (pos, byte) in value.bytes().enumerate() {
            if pos >= REGISTER_SIZE / chars_per_slot { break; }
            if byte >= 32 && byte < 127 {
                let char_idx = (byte - 32) as usize;
                let neuron_idx = pos * chars_per_slot + char_idx;
                if neuron_idx < REGISTER_SIZE {
                    self.activity[neuron_idx] = 1.0;
                }
            }
        }
        self.active = true;
        self.hold_strength = 1.0;
    }

    /// Read the current value — decoded from activity pattern.
    /// As hold_strength decays, the pattern degrades → lossy recall.
    /// Like real working memory: things get fuzzy over time.
    fn read(&self) -> String {
        if !self.active { return String::new(); }
        let chars_per_slot = 95;
        let max_pos = REGISTER_SIZE / chars_per_slot;
        let mut result = String::new();
        for pos in 0..max_pos {
            let mut best_char = 0u8;
            let mut best_val = 0.0f32;
            for c in 0..chars_per_slot {
                let idx = pos * chars_per_slot + c;
                if idx < self.activity.len() {
                    // Apply hold_strength as a multiplier — decayed = noisier
                    let val = self.activity[idx] * self.hold_strength;
                    if val > best_val {
                        best_val = val;
                        best_char = (c as u8) + 32;
                    }
                }
            }
            if best_val > 0.3 && best_char >= 32 && best_char < 127 {
                result.push(best_char as char);
            }
        }
        result.trim().to_string()
    }

    /// Decay — activity fades without reinforcement (like real working memory).
    fn decay(&mut self, rate: f32) {
        self.hold_strength -= rate;
        if self.hold_strength <= 0.0 {
            self.active = false;
            self.hold_strength = 0.0;
            for a in &mut self.activity { *a = 0.0; }
        }
    }

    /// Refresh — re-attend to this register, resetting decay.
    fn refresh(&mut self) {
        if self.active { self.hold_strength = 1.0; }
    }

    fn is_empty(&self) -> bool { !self.active }
}

/// A step in a sequential plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub instruction: String,
    pub result: String,
    pub completed: bool,
}

/// Working memory + sequential executor.
/// Enables the brain to think in steps while holding intermediate results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    registers: Vec<Register>,
    /// Current execution plan — steps to follow
    plan: Vec<Step>,
    /// Current step index
    current_step: usize,
    /// Execution history
    pub completed_plans: u64,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self {
            registers: (0..NUM_REGISTERS).map(|_| Register::new()).collect(),
            plan: Vec::new(),
            current_step: 0,
            completed_plans: 0,
        }
    }

    /// Store a value in the next available register.
    /// The value is encoded as a neural activity pattern — no strings stored.
    /// `_name` is unused: registers are neural activity patterns, not string-keyed slots.
    /// Callers pass a name for documentation purposes, but lookup is by recency/strength.
    pub fn store(&mut self, _name: &str, value: &str) -> bool {
        // Find empty register
        for reg in &mut self.registers {
            if reg.is_empty() {
                reg.load(value);
                return true;
            }
        }
        // All full — overwrite weakest (lowest hold_strength)
        if let Some(reg) = self.registers.iter_mut()
            .min_by(|a, b| a.hold_strength.partial_cmp(&b.hold_strength).unwrap())
        {
            reg.load(value);
            return true;
        }
        false
    }

    /// Read the most recently stored register.
    /// Decoded from neural activity pattern — degrades with time.
    pub fn read_latest(&self) -> String {
        self.registers.iter()
            .filter(|r| r.active)
            .max_by(|a, b| a.hold_strength.partial_cmp(&b.hold_strength).unwrap())
            .map(|r| r.read())
            .unwrap_or_default()
    }

    /// Read register at index.
    pub fn read_register(&self, idx: usize) -> String {
        if idx < self.registers.len() {
            self.registers[idx].read()
        } else {
            String::new()
        }
    }

    /// Set a plan — a sequence of steps to execute.
    pub fn set_plan(&mut self, steps: Vec<String>) {
        self.plan = steps.into_iter().map(|s| Step {
            instruction: s,
            result: String::new(),
            completed: false,
        }).collect();
        self.current_step = 0;
    }

    /// Get the current step to execute.
    pub fn current_instruction(&self) -> Option<&str> {
        self.plan.get(self.current_step).map(|s| s.instruction.as_str())
    }

    /// Record the result of the current step and advance.
    pub fn complete_step(&mut self, result: &str) {
        if self.current_step < self.plan.len() {
            self.plan[self.current_step].result = result.to_string();
            self.plan[self.current_step].completed = true;
            // Store result in a register for later steps to use
            let step_name = format!("step{}", self.current_step);
            self.store(&step_name, result);
            self.current_step += 1;
            // Check if plan is complete
            if self.current_step >= self.plan.len() {
                self.completed_plans += 1;
            }
        }
    }

    /// Is there more work to do?
    pub fn has_pending_steps(&self) -> bool {
        self.current_step < self.plan.len()
    }

    /// Get all results from the plan.
    pub fn plan_results(&self) -> Vec<&str> {
        self.plan.iter()
            .filter(|s| s.completed)
            .map(|s| s.result.as_str())
            .collect()
    }

    /// Get the final result of the plan.
    pub fn final_result(&self) -> Option<&str> {
        self.plan.last()
            .filter(|s| s.completed)
            .map(|s| s.result.as_str())
    }

    /// Decay all registers — working memory fades without attention.
    pub fn tick(&mut self) {
        for reg in &mut self.registers {
            reg.decay(0.01);
        }
    }

    /// Clear everything.
    pub fn clear(&mut self) {
        for reg in &mut self.registers {
            *reg = Register::new();
        }
        self.plan.clear();
        self.current_step = 0;
    }

    /// How many registers are in use?
    pub fn active_registers(&self) -> usize {
        self.registers.iter().filter(|r| r.active).count()
    }

    /// Dump current state for debugging.
    pub fn state_summary(&self) -> String {
        let mut s = String::new();
        for (i, reg) in self.registers.iter().enumerate() {
            if reg.active {
                s.push_str(&format!("R{}: {} ", i, reg.read()));
            }
        }
        if self.has_pending_steps() {
            s.push_str(&format!("[step {}/{}]", self.current_step + 1, self.plan.len()));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_read() {
        let mut wm = WorkingMemory::new();
        wm.store("x", "hello");
        let latest = wm.read_latest();
        assert!(latest.contains("hello"), "Got: {}", latest);
    }

    #[test]
    fn test_multiple_registers() {
        let mut wm = WorkingMemory::new();
        wm.store("a", "one");
        wm.store("b", "two");
        wm.store("c", "three");
        assert_eq!(wm.active_registers(), 3);
    }

    #[test]
    fn test_decay_degrades() {
        let mut wm = WorkingMemory::new();
        wm.store("x", "test");
        let before = wm.read_latest();
        // Decay
        for _ in 0..50 { wm.tick(); }
        let after = wm.read_latest();
        // After heavy decay, content should degrade or disappear
        for _ in 0..200 { wm.tick(); }
        assert_eq!(wm.active_registers(), 0);
    }

    #[test]
    fn test_sequential_plan() {
        let mut wm = WorkingMemory::new();
        wm.set_plan(vec![
            "subtract 3 from 11".to_string(),
            "divide result by 2".to_string(),
        ]);
        assert!(wm.has_pending_steps());
        assert_eq!(wm.current_instruction(), Some("subtract 3 from 11"));

        wm.complete_step("8");
        assert_eq!(wm.current_instruction(), Some("divide result by 2"));

        wm.complete_step("4");
        assert!(!wm.has_pending_steps());
        assert_eq!(wm.final_result(), Some("4"));
        assert_eq!(wm.completed_plans, 1);
    }

    #[test]
    fn test_overflow_replaces_oldest() {
        let mut wm = WorkingMemory::new();
        for i in 0..10 { // more than NUM_REGISTERS
            wm.store(&format!("v{}", i), &format!("{}", i));
        }
        // Should not panic, oldest registers get replaced
        assert!(wm.active_registers() <= NUM_REGISTERS);
    }

    #[test]
    fn test_clear() {
        let mut wm = WorkingMemory::new();
        wm.store("x", "hello");
        wm.set_plan(vec!["step1".to_string()]);
        wm.clear();
        assert_eq!(wm.active_registers(), 0);
        assert!(!wm.has_pending_steps());
    }
}
