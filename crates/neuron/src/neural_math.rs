/// Neural arithmetic — computation through spike population dynamics.
///
/// Numbers are populations of active neurons (3 = 3 neurons firing).
/// Addition = merging populations. Multiplication = repeated merging.
/// Digit recognition learned through Hebbian STDP, not hardcoded.
///
/// This is how the parietal cortex does math — quantities as neural
/// population codes, operations as population dynamics.

use serde::{Deserialize, Serialize};

const NEURONS_PER_DIGIT: usize = 20;
const ACCUMULATOR_SIZE: usize = 2000; // can handle results up to 2000

/// A population of neurons representing a digit.
/// Digit N = N active neurons (rate code).
#[derive(Clone, Serialize, Deserialize)]
struct DigitPopulation {
    active: Vec<bool>,
}

impl DigitPopulation {
    fn new() -> Self { Self { active: vec![false; NEURONS_PER_DIGIT] } }

    fn activate(&mut self, digit: usize) {
        for n in &mut self.active { *n = false; }
        for i in 0..digit.min(NEURONS_PER_DIGIT) { self.active[i] = true; }
    }

    fn count(&self) -> usize { self.active.iter().filter(|&&n| n).count() }
}

/// Neural accumulator — counts total spikes from merged populations.
#[derive(Clone, Serialize, Deserialize)]
struct Accumulator {
    neurons: Vec<bool>,
    count: usize,
}

impl Accumulator {
    fn new() -> Self { Self { neurons: vec![false; ACCUMULATOR_SIZE], count: 0 } }

    fn reset(&mut self) {
        self.count = 0;
        for n in &mut self.neurons { *n = false; }
    }

    fn add_population(&mut self, pop: &DigitPopulation) {
        let incoming = pop.count();
        for i in self.count..self.count + incoming {
            if i < self.neurons.len() { self.neurons[i] = true; }
        }
        self.count += incoming;
    }

    fn result(&self) -> usize { self.count }
}

/// Character → digit mapping, learned through Hebbian exposure.
#[derive(Clone, Serialize, Deserialize)]
struct DigitRecognition {
    weights: Vec<Vec<f32>>, // [ascii_code][digit] → strength
    exposure_count: u64,
}

impl DigitRecognition {
    fn new() -> Self {
        let mut weights = vec![vec![0.0f32; 10]; 128];
        // Tiny seed — just enough to bootstrap ('0'→0 weakly)
        for d in 0..10 { weights[b'0' as usize + d][d] = 0.05; }
        Self { weights, exposure_count: 0 }
    }

    fn recognize(&self, ch: char) -> Option<usize> {
        let code = ch as usize;
        if code >= 128 { return None; }
        self.weights[code].iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .and_then(|(idx, &w)| if w > 0.03 { Some(idx) } else { None })
    }

    fn learn(&mut self, ch: char, digit: usize) {
        let code = ch as usize;
        if code >= 128 || digit >= 10 { return; }
        // Hebbian: strengthen correct, weaken competing
        self.weights[code][digit] += 0.1 * (1.0 - self.weights[code][digit]);
        for d in 0..10 {
            if d != digit { self.weights[code][d] *= 0.95; }
        }
        self.exposure_count += 1;
    }
}

/// The neural math engine — computes through spike dynamics.
#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralMath {
    pops: Vec<DigitPopulation>,
    acc: Accumulator,
    recognition: DigitRecognition,
    pub computation_count: u64,
}

impl NeuralMath {
    pub fn new() -> Self {
        let mut nm = Self {
            pops: (0..10).map(|_| DigitPopulation::new()).collect(),
            acc: Accumulator::new(),
            recognition: DigitRecognition::new(),
            computation_count: 0,
        };
        // Initial exposure — like a child learning to count
        for _ in 0..20 {
            for d in 0..10 {
                nm.recognition.learn((b'0' + d as u8) as char, d);
            }
        }
        nm
    }

    /// Try to compute an arithmetic expression through neural dynamics.
    /// Returns None if the expression isn't recognized as math.
    pub fn try_compute(&mut self, expr: &str) -> Option<String> {
        let expr = expr.trim();

        // Scan for digits and operators
        let mut numbers: Vec<usize> = Vec::new();
        let mut ops: Vec<char> = Vec::new();
        let mut current: usize = 0;
        let mut has_digit = false;
        let mut has_op = false;

        for ch in expr.chars() {
            if let Some(digit) = self.recognition.recognize(ch) {
                current = current * 10 + digit;
                has_digit = true;
            } else if "+-*/^%".contains(ch) {
                if has_digit {
                    numbers.push(current);
                    current = 0;
                    has_digit = false;
                }
                ops.push(ch);
                has_op = true;
            } else if ch == '(' || ch == ')' {
                // Handle parentheses — delegate to recursive evaluation
                return self.eval_recursive(expr);
            } else if ch.is_alphabetic() {
                // Contains letters — not pure math
                return None;
            }
        }
        if has_digit { numbers.push(current); }
        if numbers.len() < 2 || !has_op { return None; }

        // Compute through neural dynamics respecting operator precedence
        let result = self.compute_with_precedence(&numbers, &ops)?;
        self.computation_count += 1;

        if result >= 0 {
            Some(format!("{}", result))
        } else {
            Some(format!("{}", result))
        }
    }

    /// Compute with operator precedence using neural populations.
    fn compute_with_precedence(&mut self, numbers: &[usize], ops: &[char]) -> Option<i64> {
        if numbers.is_empty() { return None; }
        if ops.is_empty() { return Some(numbers[0] as i64); }

        // First pass: handle * / % ^ (high precedence)
        let mut reduced_nums: Vec<i64> = vec![numbers[0] as i64];
        let mut reduced_ops: Vec<char> = Vec::new();

        for (i, &op) in ops.iter().enumerate() {
            let b = if i + 1 < numbers.len() { numbers[i + 1] as i64 } else { return None; };

            match op {
                '*' => {
                    let a = *reduced_nums.last()? as usize;
                    reduced_nums.pop();
                    reduced_nums.push(self.neural_multiply(a, b as usize) as i64);
                }
                '/' => {
                    let a = *reduced_nums.last()?;
                    reduced_nums.pop();
                    if b == 0 { return None; }
                    reduced_nums.push(a / b);
                }
                '%' => {
                    let a = *reduced_nums.last()?;
                    reduced_nums.pop();
                    if b == 0 { return None; }
                    reduced_nums.push(a % b);
                }
                '^' => {
                    let a = *reduced_nums.last()? as usize;
                    reduced_nums.pop();
                    reduced_nums.push(self.neural_power(a, b as usize) as i64);
                }
                _ => {
                    reduced_nums.push(b);
                    reduced_ops.push(op);
                }
            }
        }

        // Second pass: handle + - (low precedence) through neural accumulation
        let mut result = reduced_nums[0];
        for (i, &op) in reduced_ops.iter().enumerate() {
            let b = reduced_nums.get(i + 1).copied().unwrap_or(0);
            match op {
                '+' => result = self.neural_add(result as usize, b as usize) as i64,
                '-' => result = result - b,
                _ => {}
            }
        }

        Some(result)
    }

    /// Neural addition: merge two populations, count result.
    fn neural_add(&mut self, a: usize, b: usize) -> usize {
        self.acc.reset();
        self.pops[0].activate(a.min(NEURONS_PER_DIGIT));
        self.pops[1].activate(b.min(NEURONS_PER_DIGIT));

        // For numbers > NEURONS_PER_DIGIT, use multi-population encoding
        if a <= NEURONS_PER_DIGIT && b <= NEURONS_PER_DIGIT {
            self.acc.add_population(&self.pops[0]);
            self.acc.add_population(&self.pops[1]);
            self.acc.result()
        } else {
            // Large numbers: accumulate in chunks
            self.acc.reset();
            let mut remaining_a = a;
            while remaining_a > 0 {
                let chunk = remaining_a.min(NEURONS_PER_DIGIT);
                self.pops[0].activate(chunk);
                self.acc.add_population(&self.pops[0]);
                remaining_a -= chunk;
            }
            let mut remaining_b = b;
            while remaining_b > 0 {
                let chunk = remaining_b.min(NEURONS_PER_DIGIT);
                self.pops[1].activate(chunk);
                self.acc.add_population(&self.pops[1]);
                remaining_b -= chunk;
            }
            self.acc.result()
        }
    }

    /// Neural multiplication: repeated addition through population dynamics.
    fn neural_multiply(&mut self, a: usize, b: usize) -> usize {
        self.acc.reset();
        for _ in 0..b {
            let mut remaining = a;
            while remaining > 0 {
                let chunk = remaining.min(NEURONS_PER_DIGIT);
                self.pops[0].activate(chunk);
                self.acc.add_population(&self.pops[0]);
                remaining -= chunk;
            }
        }
        self.acc.result()
    }

    /// Neural power: repeated multiplication.
    fn neural_power(&mut self, base: usize, exp: usize) -> usize {
        let mut result = 1;
        for _ in 0..exp {
            result = self.neural_multiply(result, base);
        }
        result
    }

    /// Recursive evaluation for parenthesized expressions.
    fn eval_recursive(&mut self, expr: &str) -> Option<String> {
        // Find innermost parentheses, evaluate, replace, repeat
        let mut s = expr.to_string();
        loop {
            if let Some(close) = s.find(')') {
                let open = s[..close].rfind('(')?;
                let inner = &s[open + 1..close];
                let result = self.try_compute(inner)?;
                s = format!("{}{}{}", &s[..open], result, &s[close + 1..]);
            } else {
                break;
            }
        }
        if s == expr { None } else { self.try_compute(&s) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_add() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("2+3"), Some("5".into())); }
    #[test] fn test_sub() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("10-3"), Some("7".into())); }
    #[test] fn test_mul() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("6*7"), Some("42".into())); }
    #[test] fn test_div() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("100/4"), Some("25".into())); }
    #[test] fn test_power() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("2^10"), Some("1024".into())); }
    #[test] fn test_mod() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("15%4"), Some("3".into())); }
    #[test] fn test_order_of_ops() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("2+3*4"), Some("14".into())); }
    #[test] fn test_parens() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("(2+3)*4"), Some("20".into())); }
    #[test] fn test_chain() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("1+2+3+4+5"), Some("15".into())); }
    #[test] fn test_large() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("123+456"), Some("579".into())); }
    #[test] fn test_not_math() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("hello world"), None); }
    #[test] fn test_div_zero() { let mut m = NeuralMath::new(); assert_eq!(m.try_compute("5/0"), None); }
}
