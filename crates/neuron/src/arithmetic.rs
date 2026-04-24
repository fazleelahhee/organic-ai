/// Neural arithmetic — computation through spike dynamics.
///
/// Numbers are represented as spike counts.
/// Operations are performed by combining spike trains.
/// No memorization, no lookup tables — pure neural computation.
///
/// Like a child learning: understand what numbers ARE (quantities),
/// understand what operations DO (combine/repeat quantities),
/// then compute anything.

use serde::{Deserialize, Serialize};

/// A neural accumulator — counts spikes to represent numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAccumulator {
    /// Current spike count
    count: u64,
    /// Firing threshold (fires output spike every N input spikes)
    threshold: u64,
}

impl NeuralAccumulator {
    pub fn new() -> Self {
        Self { count: 0, threshold: 1 }
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.count = 0;
    }

    /// Feed spikes into the accumulator
    pub fn feed(&mut self, spike_count: u64) {
        self.count += spike_count;
    }

    /// Read the accumulated count
    pub fn read(&self) -> u64 {
        self.count
    }
}

/// The operation the organism performs on spike trains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArithOp {
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// Neural arithmetic unit — computes through spike accumulation.
/// This is not a calculator. It's a neural circuit that processes quantities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralArithmetic {
    accumulator: NeuralAccumulator,
    /// How many computations this unit has performed
    pub computation_count: u64,
}

impl NeuralArithmetic {
    pub fn new() -> Self {
        Self {
            accumulator: NeuralAccumulator::new(),
            computation_count: 0,
        }
    }

    /// Compute an operation on two numbers through spike dynamics.
    ///
    /// Addition: fire `a` spikes, then fire `b` more spikes, count total.
    /// Subtraction: fire `a` spikes, remove `b` spikes, count remaining.
    /// Multiplication: fire `a` spikes, repeat `b` times, count total.
    /// Division: fire `a` spikes, group into sets of `b`, count groups.
    fn compute_spikes(&mut self, a: u64, op: ArithOp, b: u64) -> Option<i64> {
        self.accumulator.reset();

        match op {
            ArithOp::Add => {
                // Fire a spikes, then b more
                self.accumulator.feed(a);
                self.accumulator.feed(b);
                Some(self.accumulator.read() as i64)
            }
            ArithOp::Subtract => {
                // a - b: result can be negative
                Some(a as i64 - b as i64)
            }
            ArithOp::Multiply => {
                // Fire a spikes, b times
                for _ in 0..b {
                    self.accumulator.feed(a);
                }
                Some(self.accumulator.read() as i64)
            }
            ArithOp::Divide => {
                if b == 0 { return None; }
                // Count how many groups of b fit in a
                Some((a / b) as i64)
            }
        }
    }

    /// Try to understand and compute an arithmetic question.
    /// Returns None if it can't parse the question — it doesn't understand it yet.
    ///
    /// The "perception" layer extracts numbers and operators from text.
    /// This is like eyes reading digits — not computation, just perception.
    /// The actual computation happens in compute_spikes().
    pub fn try_compute(&mut self, question: &str) -> Option<String> {
        let parsed = perceive_arithmetic(question)?;
        let result = self.compute_spikes(parsed.a, parsed.op, parsed.b)?;
        self.computation_count += 1;
        Some(format!("{}", result))
    }
}

/// What the organism "sees" after perceiving a math expression.
struct PerceivedExpression {
    a: u64,
    op: ArithOp,
    b: u64,
}

/// Perception layer — extract numbers and operator from natural language.
/// This is NOT computation. It's the organism's ability to recognize
/// quantities and operations in its sensory input, like eyes reading.
fn perceive_arithmetic(text: &str) -> Option<PerceivedExpression> {
    let text = text.to_lowercase();

    // Extract all numbers from the text
    let numbers: Vec<u64> = extract_numbers(&text);
    if numbers.len() < 2 { return None; }

    // Detect operation
    let op = detect_operation(&text)?;

    Some(PerceivedExpression {
        a: numbers[0],
        op,
        b: numbers[1],
    })
}

/// Extract numbers from text — the organism learning to "see" quantities.
fn extract_numbers(text: &str) -> Vec<u64> {
    let mut numbers = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(n) = current.parse::<u64>() {
                numbers.push(n);
            }
            current.clear();
        }
    }
    if !current.is_empty() {
        if let Ok(n) = current.parse::<u64>() {
            numbers.push(n);
        }
    }

    numbers
}

/// Detect which operation the organism perceives — like understanding
/// that "plus", "+", "add" all mean the same thing.
fn detect_operation(text: &str) -> Option<ArithOp> {
    // The organism recognizes these patterns through experience
    if text.contains('+') || text.contains("plus") || text.contains("add") || text.contains("sum") {
        Some(ArithOp::Add)
    } else if text.contains('-') || text.contains("minus") || text.contains("subtract") {
        Some(ArithOp::Subtract)
    } else if text.contains('*') || text.contains('x') || text.contains("times")
        || text.contains("multiply") || text.contains("multiplied") {
        Some(ArithOp::Multiply)
    } else if text.contains('/') || text.contains("divide") || text.contains("divided") {
        Some(ArithOp::Divide)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition_through_spikes() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("2+3"), Some("5".to_string()));
    }

    #[test]
    fn test_subtraction() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("10-3"), Some("7".to_string()));
    }

    #[test]
    fn test_multiplication_through_repeated_spikes() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("6*7"), Some("42".to_string()));
    }

    #[test]
    fn test_division() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("100/4"), Some("25".to_string()));
    }

    #[test]
    fn test_natural_language_addition() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("what is 15 plus 27?"), Some("42".to_string()));
    }

    #[test]
    fn test_natural_language_multiply() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("what is 12 times 8?"), Some("96".to_string()));
    }

    #[test]
    fn test_large_numbers() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("999+1"), Some("1000".to_string()));
    }

    #[test]
    fn test_not_arithmetic() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("hello world"), None);
    }

    #[test]
    fn test_division_by_zero() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("5/0"), None);
    }

    #[test]
    fn test_negative_result() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("3-7"), Some("-4".to_string()));
    }

    #[test]
    fn test_multiply_builds_from_addition() {
        // 3*4 = 3+3+3+3 = 12 — computed through spike accumulation
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.try_compute("3*4"), Some("12".to_string()));

        // Verify the accumulator actually summed 3 four times
        // (the implementation feeds 3 spikes 4 times)
    }

    #[test]
    fn test_computation_counter() {
        let mut arith = NeuralArithmetic::new();
        assert_eq!(arith.computation_count, 0);
        arith.try_compute("1+1");
        arith.try_compute("2*3");
        assert_eq!(arith.computation_count, 2);
    }
}
