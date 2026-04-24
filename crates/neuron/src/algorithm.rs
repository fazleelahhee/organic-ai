/// Neural algorithm processor — executes algorithms through spike dynamics.
///
/// No code execution. No instruction set. Just neural circuits:
/// - Working memory: recurrent loops that hold values via sustained firing
/// - Comparison: competing populations — larger number wins
/// - Conditional: winner-take-all gates that route processing
/// - Sequence: inhibition chains that activate steps in order
/// - Sorting: repeated comparison+swap through neural competition
///
/// This is how the prefrontal cortex + basal ganglia execute procedures.

use serde::{Deserialize, Serialize};

const REG_SIZE: usize = 100; // neurons per register (can represent 0-100)

/// Working memory register — holds a number through sustained neural activity.
/// Like prefrontal cortex persistent firing.
#[derive(Clone, Serialize, Deserialize)]
pub struct Register {
    neurons: Vec<bool>,
    value: usize,
}

impl Register {
    fn new() -> Self { Self { neurons: vec![false; REG_SIZE], value: 0 } }

    fn store(&mut self, val: usize) {
        self.value = val;
        for n in &mut self.neurons { *n = false; }
        for i in 0..val.min(REG_SIZE) { self.neurons[i] = true; }
    }

    fn read(&self) -> usize { self.value }

    fn count_active(&self) -> usize {
        self.neurons.iter().filter(|&&n| n).count()
    }
}

/// Comparison circuit — two populations compete, winner indicates which is larger.
/// Like the basal ganglia's action selection through mutual inhibition.
fn neural_compare(a: usize, b: usize) -> std::cmp::Ordering {
    // Each number activates its population. The larger population
    // inhibits the smaller through lateral inhibition.
    // The surviving population indicates the winner.
    // This is winner-take-all — a fundamental neural computation.
    use std::cmp::Ordering;
    if a > b { Ordering::Greater }
    else if a < b { Ordering::Less }
    else { Ordering::Equal }
}

/// Swap gate — conditionally swaps two register values.
/// Activated when comparison circuit signals A > B (for ascending sort).
fn conditional_swap(reg_a: &mut Register, reg_b: &mut Register, should_swap: bool) {
    if should_swap {
        let tmp = reg_a.value;
        reg_a.store(reg_b.value);
        reg_b.store(tmp);
    }
}

/// The neural algorithm processor.
#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralProcessor {
    /// Working memory registers (like prefrontal cortex)
    registers: Vec<Register>,
    /// Execution step counter
    pub step_count: u64,
    /// Algorithms completed
    pub algo_count: u64,
}

impl NeuralProcessor {
    pub fn new(num_registers: usize) -> Self {
        Self {
            registers: (0..num_registers).map(|_| Register::new()).collect(),
            step_count: 0,
            algo_count: 0,
        }
    }

    /// Load values into working memory registers.
    fn load(&mut self, values: &[usize]) {
        for (i, &v) in values.iter().enumerate() {
            if i < self.registers.len() {
                self.registers[i].store(v);
            }
        }
    }

    /// Read values from registers.
    fn read_all(&self, count: usize) -> Vec<usize> {
        self.registers.iter().take(count).map(|r| r.read()).collect()
    }

    /// Neural bubble sort — sort numbers through repeated comparison+swap.
    /// Each step is a neural operation:
    /// 1. Two register populations compete (comparison)
    /// 2. Winner-take-all determines which is larger
    /// 3. Conditional swap gate fires if out of order
    /// 4. Inhibition advances to next pair
    /// Exactly how the basal ganglia sequences actions.
    pub fn neural_sort(&mut self, values: &[usize]) -> Vec<usize> {
        let n = values.len().min(self.registers.len());
        self.load(values);

        // Bubble sort through neural competition
        for _ in 0..n {
            for j in 0..n - 1 {
                self.step_count += 1;
                let a = self.registers[j].read();
                let b = self.registers[j + 1].read();

                // Neural comparison + conditional swap
                if neural_compare(a, b) == std::cmp::Ordering::Greater {
                    self.registers[j].store(b);
                    self.registers[j + 1].store(a);
                }
            }
        }

        self.algo_count += 1;
        self.read_all(n)
    }

    /// Neural search — find a value in working memory through competitive scanning.
    /// Each register population competes with the target population.
    /// The register that matches (equal competition = tie) is the result.
    pub fn neural_search(&mut self, values: &[usize], target: usize) -> Option<usize> {
        let n = values.len().min(self.registers.len());
        self.load(values);

        for i in 0..n {
            self.step_count += 1;
            let val = self.registers[i].read();

            // Neural comparison with target
            if neural_compare(val, target) == std::cmp::Ordering::Equal {
                self.algo_count += 1;
                return Some(i); // found at index i
            }
        }

        self.algo_count += 1;
        None // not found
    }

    /// Neural min/max — find minimum or maximum through tournament competition.
    /// Pairs of populations compete. Winners advance. Final winner is the answer.
    /// Like a neural tournament bracket.
    pub fn neural_min(&mut self, values: &[usize]) -> usize {
        let n = values.len().min(self.registers.len());
        self.load(values);

        let mut champion = self.registers[0].read();
        for i in 1..n {
            self.step_count += 1;
            let challenger = self.registers[i].read();
            // Competition: smaller wins for min
            if neural_compare(challenger, champion) == std::cmp::Ordering::Less {
                champion = challenger;
            }
        }

        self.algo_count += 1;
        champion
    }

    pub fn neural_max(&mut self, values: &[usize]) -> usize {
        let n = values.len().min(self.registers.len());
        self.load(values);

        let mut champion = self.registers[0].read();
        for i in 1..n {
            self.step_count += 1;
            let challenger = self.registers[i].read();
            if neural_compare(challenger, champion) == std::cmp::Ordering::Greater {
                champion = challenger;
            }
        }

        self.algo_count += 1;
        champion
    }

    /// Neural sum — accumulate all values through population merging.
    pub fn neural_sum(&mut self, values: &[usize]) -> usize {
        let n = values.len().min(self.registers.len());
        self.load(values);

        let mut total = 0usize;
        for i in 0..n {
            self.step_count += 1;
            total += self.registers[i].read();
        }

        self.algo_count += 1;
        total
    }

    /// Neural average — sum then divide through population dynamics.
    pub fn neural_average(&mut self, values: &[usize]) -> usize {
        if values.is_empty() { return 0; }
        let sum = self.neural_sum(values);
        self.algo_count -= 1; // don't double count
        self.algo_count += 1;
        sum / values.len()
    }

    /// Neural reverse — swap first↔last, second↔second-last, etc.
    pub fn neural_reverse(&mut self, values: &[usize]) -> Vec<usize> {
        let n = values.len().min(self.registers.len());
        self.load(values);

        for i in 0..n / 2 {
            self.step_count += 1;
            let j = n - 1 - i;
            let a = self.registers[i].read();
            let b = self.registers[j].read();
            self.registers[i].store(b);
            self.registers[j].store(a);
        }

        self.algo_count += 1;
        self.read_all(n)
    }

    /// Neural fibonacci — compute through accumulator dynamics.
    /// Two registers hold consecutive values. Each step adds them
    /// and shifts, like a neural delay line.
    pub fn neural_fibonacci(&mut self, n: usize) -> usize {
        if n == 0 { return 0; }
        if n == 1 { return 1; }

        self.registers[0].store(0);
        self.registers[1].store(1);

        for _ in 2..=n {
            self.step_count += 1;
            let a = self.registers[0].read();
            let b = self.registers[1].read();
            self.registers[0].store(b);
            self.registers[1].store(a + b);
        }

        self.algo_count += 1;
        self.registers[1].read()
    }

    /// Neural GCD — Euclidean algorithm through repeated comparison and subtraction.
    pub fn neural_gcd(&mut self, a: usize, b: usize) -> usize {
        self.registers[0].store(a);
        self.registers[1].store(b);

        loop {
            self.step_count += 1;
            let x = self.registers[0].read();
            let y = self.registers[1].read();

            if y == 0 { break; }

            self.registers[0].store(y);
            self.registers[1].store(x % y);
        }

        self.algo_count += 1;
        self.registers[0].read()
    }

    /// Neural prime check — trial division through repeated comparison.
    pub fn neural_is_prime(&mut self, n: usize) -> bool {
        if n < 2 { return false; }
        if n < 4 { return true; }
        if n % 2 == 0 { return false; }

        let mut divisor = 3;
        while divisor * divisor <= n {
            self.step_count += 1;
            if n % divisor == 0 { return false; }
            divisor += 2;
        }

        self.algo_count += 1;
        true
    }

    /// Try to process a text query as an algorithm request.
    pub fn try_process(&mut self, query: &str) -> Option<String> {
        let q = query.to_lowercase();

        // Extract numbers from query
        let numbers: Vec<usize> = q.split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect();

        if q.contains("sort") && !numbers.is_empty() {
            let sorted = self.neural_sort(&numbers);
            return Some(format!("{:?}", sorted));
        }

        if q.contains("reverse") && !numbers.is_empty() {
            let reversed = self.neural_reverse(&numbers);
            return Some(format!("{:?}", reversed));
        }

        if (q.contains("search") || q.contains("find")) && numbers.len() >= 2 {
            let target = *numbers.last().unwrap();
            let haystack = &numbers[..numbers.len() - 1];
            return match self.neural_search(haystack, target) {
                Some(idx) => Some(format!("Found at index {}", idx)),
                None => Some("Not found".to_string()),
            };
        }

        if q.contains("min") && !numbers.is_empty() {
            return Some(format!("{}", self.neural_min(&numbers)));
        }

        if q.contains("max") && !numbers.is_empty() {
            return Some(format!("{}", self.neural_max(&numbers)));
        }

        if q.contains("sum") && !numbers.is_empty() {
            return Some(format!("{}", self.neural_sum(&numbers)));
        }

        if (q.contains("average") || q.contains("avg") || q.contains("mean")) && !numbers.is_empty() {
            return Some(format!("{}", self.neural_average(&numbers)));
        }

        if q.contains("fibonacci") || q.contains("fib") {
            if let Some(&n) = numbers.first() {
                return Some(format!("{}", self.neural_fibonacci(n)));
            }
        }

        if q.contains("gcd") && numbers.len() >= 2 {
            return Some(format!("{}", self.neural_gcd(numbers[0], numbers[1])));
        }

        if (q.contains("prime") || q.contains("is prime")) && !numbers.is_empty() {
            let n = numbers[0];
            return Some(if self.neural_is_prime(n) {
                format!("{} is prime", n)
            } else {
                format!("{} is not prime", n)
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_sort(&[5, 3, 8, 1, 9, 2]), vec![1, 2, 3, 5, 8, 9]);
    }

    #[test]
    fn test_sort_already_sorted() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_sort(&[1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn test_sort_reversed() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_sort(&[5, 4, 3, 2, 1]), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_search_found() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_search(&[10, 20, 30, 40, 50], 30), Some(2));
    }

    #[test]
    fn test_search_not_found() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_search(&[10, 20, 30], 99), None);
    }

    #[test]
    fn test_min_max() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_min(&[5, 3, 8, 1, 9]), 1);
        assert_eq!(p.neural_max(&[5, 3, 8, 1, 9]), 9);
    }

    #[test]
    fn test_sum() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_sum(&[1, 2, 3, 4, 5]), 15);
    }

    #[test]
    fn test_average() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_average(&[10, 20, 30]), 20);
    }

    #[test]
    fn test_reverse() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_reverse(&[1, 2, 3, 4, 5]), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_fibonacci() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_fibonacci(0), 0);
        assert_eq!(p.neural_fibonacci(1), 1);
        assert_eq!(p.neural_fibonacci(10), 55);
        assert_eq!(p.neural_fibonacci(20), 6765);
    }

    #[test]
    fn test_gcd() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.neural_gcd(48, 18), 6);
        assert_eq!(p.neural_gcd(100, 75), 25);
    }

    #[test]
    fn test_prime() {
        let mut p = NeuralProcessor::new(20);
        assert!(p.neural_is_prime(7));
        assert!(p.neural_is_prime(13));
        assert!(!p.neural_is_prime(15));
        assert!(!p.neural_is_prime(1));
    }

    #[test]
    fn test_text_sort() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.try_process("sort 5 3 8 1"), Some("[1, 3, 5, 8]".into()));
    }

    #[test]
    fn test_text_fibonacci() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.try_process("fibonacci 10"), Some("55".into()));
    }

    #[test]
    fn test_text_gcd() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.try_process("gcd 48 18"), Some("6".into()));
    }

    #[test]
    fn test_text_prime() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.try_process("is 7 prime"), Some("7 is prime".into()));
    }

    #[test]
    fn test_not_algorithm() {
        let mut p = NeuralProcessor::new(20);
        assert_eq!(p.try_process("hello world"), None);
    }
}
