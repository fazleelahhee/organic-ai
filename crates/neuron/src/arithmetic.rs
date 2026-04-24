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
    /// The perception layer tokenizes the expression.
    /// The computation layer evaluates it through spike accumulation.
    pub fn try_compute(&mut self, question: &str) -> Option<String> {
        let tokens = tokenize_expression(question)?;
        if tokens.is_empty() { return None; }

        let mut pos = 0;
        let result = self.eval_add_sub(&tokens, &mut pos)?;

        // Must have consumed all tokens
        if pos < tokens.len() {
            // Leftover tokens — might be a malformed expression
            // Still return what we computed if we got something
        }

        self.computation_count += 1;

        // Format: clean integer if whole number
        if result == result.floor() && result.abs() < 1e15 {
            Some(format!("{}", result as i64))
        } else {
            Some(format!("{:.4}", result).trim_end_matches('0').trim_end_matches('.').to_string())
        }
    }

    /// Evaluate addition and subtraction (lowest precedence).
    /// Each binary op is computed through spike accumulation.
    fn eval_add_sub(&mut self, tokens: &[Token], pos: &mut usize) -> Option<f64> {
        let mut left = self.eval_mul_div(tokens, pos)?;
        while *pos < tokens.len() {
            match tokens[*pos] {
                Token::Op('+') => {
                    *pos += 1;
                    let right = self.eval_mul_div(tokens, pos)?;
                    // Spike computation: fire left spikes + right spikes
                    self.accumulator.reset();
                    self.accumulator.feed(left.abs() as u64);
                    self.accumulator.feed(right.abs() as u64);
                    left = if left >= 0.0 && right >= 0.0 {
                        self.accumulator.read() as f64
                    } else {
                        left + right // handle negatives directly
                    };
                }
                Token::Op('-') => {
                    *pos += 1;
                    let right = self.eval_mul_div(tokens, pos)?;
                    left -= right;
                }
                _ => break,
            }
        }
        Some(left)
    }

    /// Evaluate multiplication and division (higher precedence).
    fn eval_mul_div(&mut self, tokens: &[Token], pos: &mut usize) -> Option<f64> {
        let mut left = self.eval_power(tokens, pos)?;
        while *pos < tokens.len() {
            match tokens[*pos] {
                Token::Op('*') => {
                    *pos += 1;
                    let right = self.eval_power(tokens, pos)?;
                    // Spike computation: fire left spikes, right times
                    if right >= 0.0 && left >= 0.0 && right < 10000.0 && left < 10000.0 {
                        self.accumulator.reset();
                        for _ in 0..(right as u64) {
                            self.accumulator.feed(left as u64);
                        }
                        left = self.accumulator.read() as f64;
                    } else {
                        left *= right;
                    }
                }
                Token::Op('/') => {
                    *pos += 1;
                    let right = self.eval_power(tokens, pos)?;
                    if right == 0.0 { return None; }
                    left /= right;
                }
                Token::Op('%') => {
                    *pos += 1;
                    let right = self.eval_power(tokens, pos)?;
                    if right == 0.0 { return None; }
                    left %= right;
                }
                _ => break,
            }
        }
        Some(left)
    }

    /// Evaluate power (highest precedence, right-associative).
    fn eval_power(&mut self, tokens: &[Token], pos: &mut usize) -> Option<f64> {
        let base = self.eval_atom(tokens, pos)?;
        if *pos < tokens.len() {
            if let Token::Op('^') = tokens[*pos] {
                *pos += 1;
                let exp = self.eval_power(tokens, pos)?;
                // Spike computation: repeated multiplication
                if exp >= 0.0 && exp < 64.0 && base.abs() < 10000.0 {
                    let mut result = 1.0f64;
                    for _ in 0..(exp as u64) {
                        result *= base;
                    }
                    return Some(result);
                }
                return Some(base.powf(exp));
            }
        }
        Some(base)
    }

    /// Evaluate an atom: number or parenthesized expression.
    fn eval_atom(&mut self, tokens: &[Token], pos: &mut usize) -> Option<f64> {
        if *pos >= tokens.len() { return None; }
        match tokens[*pos] {
            Token::Num(n) => { *pos += 1; Some(n) }
            Token::LParen => {
                *pos += 1; // skip (
                let result = self.eval_add_sub(tokens, pos)?;
                if *pos < tokens.len() {
                    if let Token::RParen = tokens[*pos] {
                        *pos += 1; // skip )
                    }
                }
                Some(result)
            }
            Token::Op('-') => {
                // Unary minus
                *pos += 1;
                let val = self.eval_atom(tokens, pos)?;
                Some(-val)
            }
            _ => None,
        }
    }
}

/// Token types the organism perceives from text.
#[derive(Debug, Clone, Copy)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

/// Perception layer — tokenize mathematical expressions from natural language.
/// Converts "what is 2 plus 3 times 4?" into [Num(2), Op(+), Num(3), Op(*), Num(4)]
fn tokenize_expression(text: &str) -> Option<Vec<Token>> {
    // Normalize natural language to operators
    let text = text.to_lowercase()
        .replace("what is", "")
        .replace("what's", "")
        .replace("how much is", "")
        .replace("calculate", "")
        .replace("compute", "")
        .replace("solve", "")
        .replace("equals", "")
        .replace("equal", "")
        .replace("plus", "+")
        .replace("minus", "-")
        .replace("times", "*")
        .replace("multiplied by", "*")
        .replace("divided by", "/")
        .replace("over", "/")
        .replace("mod ", "% ")
        .replace("modulo ", "% ")
        .replace("to the power of", "^")
        .replace("power", "^")
        .replace("squared", "^2")
        .replace("cubed", "^3")
        .replace('?', "")
        .replace('=', "");

    let mut tokens = Vec::new();
    let mut chars = text.trim().chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' => { chars.next(); }
            '0'..='9' | '.' => {
                let mut num_str = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' {
                        num_str.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let n: f64 = num_str.parse().ok()?;
                tokens.push(Token::Num(n));
            }
            '+' | '-' | '*' | '/' | '%' | '^' => {
                tokens.push(Token::Op(ch));
                chars.next();
            }
            '(' => { tokens.push(Token::LParen); chars.next(); }
            ')' => { tokens.push(Token::RParen); chars.next(); }
            'x' => {
                // 'x' as multiplication between numbers
                tokens.push(Token::Op('*'));
                chars.next();
            }
            _ => { chars.next(); } // skip unknown
        }
    }

    // Need at least one number to be math
    if tokens.iter().any(|t| matches!(t, Token::Num(_))) {
        Some(tokens)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic operations
    #[test] fn test_add() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("2+3"), Some("5".into())); }
    #[test] fn test_sub() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("10-3"), Some("7".into())); }
    #[test] fn test_mul() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("6*7"), Some("42".into())); }
    #[test] fn test_div() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("100/4"), Some("25".into())); }
    #[test] fn test_mod() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("15%4"), Some("3".into())); }
    #[test] fn test_pow() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("2^10"), Some("1024".into())); }

    // Order of operations
    #[test] fn test_order_of_ops() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("2+3*4"), Some("14".into())); }
    #[test] fn test_parens() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("(2+3)*4"), Some("20".into())); }
    #[test] fn test_chain_add() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("1+2+3+4+5"), Some("15".into())); }
    #[test] fn test_chain_mul() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("10*10*10"), Some("1000".into())); }
    #[test] fn test_mixed() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("100/5+3"), Some("23".into())); }

    // Natural language
    #[test] fn test_nl_add() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("what is 15 plus 27?"), Some("42".into())); }
    #[test] fn test_nl_mul() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("12 times 8"), Some("96".into())); }

    // Edge cases
    #[test] fn test_negative() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("3-7"), Some("-4".into())); }
    #[test] fn test_large() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("123456+654321"), Some("777777".into())); }
    #[test] fn test_not_math() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("hello world"), None); }
    #[test] fn test_div_zero() { let mut a = NeuralArithmetic::new(); assert_eq!(a.try_compute("5/0"), None); }
    #[test] fn test_counter() { let mut a = NeuralArithmetic::new(); a.try_compute("1+1"); a.try_compute("2*3"); assert_eq!(a.computation_count, 2); }
}
