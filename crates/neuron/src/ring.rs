/// Ring Attractor — organic number representation and computation.
///
/// Numbers are positions on a neural ring. Math emerges from dynamics:
/// - Addition = shifting the activity bump along the ring
/// - Subtraction = shifting backwards
/// - Multiplication = repeated shifting
/// - Power = repeated multiplication
///
/// No learning needed. No parser. No memorization.
/// The architecture IS the computation.
/// This is how the parietal cortex represents quantity.

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct NumberRing {
    activity: Vec<f32>,
    size: usize,
    max_value: usize,
}

impl NumberRing {
    pub fn new(size: usize, max_value: usize) -> Self {
        Self { activity: vec![0.0; size], size, max_value }
    }

    pub fn reset(&mut self) {
        for a in &mut self.activity { *a = 0.0; }
    }

    /// Inject a gaussian bump at the position representing this number.
    pub fn inject(&mut self, number: usize, strength: f32) {
        let center = (number as f32 / self.max_value as f32 * self.size as f32) as usize;
        let sigma = 3.0;
        for i in 0..self.size {
            let d1 = (i as f32 - center as f32).abs();
            let d2 = (self.size as f32 - d1).abs();
            let dist = d1.min(d2);
            self.activity[i] += strength * (-dist * dist / (2.0 * sigma * sigma)).exp();
        }
    }

    /// Shift the bump by N positions — this IS addition.
    pub fn shift_forward(&mut self, amount: usize) {
        let shift = (amount as f32 / self.max_value as f32 * self.size as f32) as usize;
        let mut new = vec![0.0f32; self.size];
        for i in 0..self.size {
            new[(i + shift) % self.size] = self.activity[i];
        }
        self.activity = new;
    }

    /// Shift backwards — this IS subtraction.
    pub fn shift_backward(&mut self, amount: usize) {
        let shift = self.size - (amount as f32 / self.max_value as f32 * self.size as f32) as usize;
        let mut new = vec![0.0f32; self.size];
        for i in 0..self.size {
            new[(i + shift) % self.size] = self.activity[i];
        }
        self.activity = new;
    }

    /// Settle dynamics — local excitation + global inhibition stabilizes bump.
    pub fn settle(&mut self, steps: usize) {
        for _ in 0..steps {
            let mut new = vec![0.0f32; self.size];
            let global_mean: f32 = self.activity.iter().sum::<f32>() / self.size as f32;
            for i in 0..self.size {
                let mut local = 0.0f32;
                for d in 1..=3 {
                    local += self.activity[(i + d) % self.size] * 0.3;
                    local += self.activity[(i + self.size - d) % self.size] * 0.3;
                }
                new[i] = self.activity[i] * 0.7 + local - global_mean * 0.01 * self.size as f32;
                if new[i] < 0.0 { new[i] = 0.0; }
            }
            self.activity = new;
        }
    }

    /// Read the peak position = the number being represented.
    pub fn read(&self) -> usize {
        let peak_idx = self.activity.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        (peak_idx as f32 / self.size as f32 * self.max_value as f32).round() as usize
    }

    /// Addition through bump dynamics.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.reset();
        self.inject(a, 1.0);
        self.settle(5);
        self.shift_forward(b);
        self.settle(5);
        self.read()
    }

    /// Subtraction through bump dynamics.
    pub fn subtract(&mut self, a: usize, b: usize) -> usize {
        self.reset();
        self.inject(a, 1.0);
        self.settle(5);
        self.shift_backward(b);
        self.settle(5);
        self.read()
    }

    /// Multiplication through repeated shifting.
    pub fn multiply(&mut self, a: usize, b: usize) -> usize {
        self.reset();
        self.inject(0, 0.1);
        self.settle(3);
        for _ in 0..b {
            self.shift_forward(a);
            self.settle(3);
        }
        self.read()
    }

    /// Power through repeated multiplication.
    pub fn power(&mut self, base: usize, exp: usize) -> usize {
        let mut result = 1;
        for _ in 0..exp {
            let current = result;
            self.reset();
            self.inject(0, 0.1);
            self.settle(2);
            for _ in 0..base {
                self.shift_forward(current);
                self.settle(2);
            }
            result = self.read();
        }
        result
    }

    /// Division through repeated subtraction.
    pub fn divide(&mut self, a: usize, b: usize) -> Option<usize> {
        if b == 0 { return None; }
        let mut count = 0;
        let mut remaining = a;
        while remaining >= b {
            remaining = self.subtract(remaining, b);
            count += 1;
            if count > self.max_value { break; }
        }
        Some(count)
    }

    /// Modulo through repeated subtraction.
    pub fn modulo(&mut self, a: usize, b: usize) -> Option<usize> {
        if b == 0 { return None; }
        let mut remaining = a;
        while remaining >= b {
            remaining = self.subtract(remaining, b);
            if remaining > a { break; } // overflow protection
        }
        Some(remaining)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let mut r = NumberRing::new(400, 100);
        assert_eq!(r.add(2, 3), 5);
        assert_eq!(r.add(10, 15), 25);
        assert_eq!(r.add(0, 7), 7);
    }

    #[test]
    fn test_subtract() {
        let mut r = NumberRing::new(400, 100);
        assert_eq!(r.subtract(10, 3), 7);
        assert_eq!(r.subtract(50, 25), 25);
    }

    #[test]
    fn test_multiply() {
        let mut r = NumberRing::new(400, 100);
        assert_eq!(r.multiply(6, 7), 42);
        assert_eq!(r.multiply(3, 4), 12);
    }

    #[test]
    fn test_power() {
        let mut r = NumberRing::new(400, 100);
        assert_eq!(r.power(2, 3), 8);
        assert_eq!(r.power(3, 2), 9);
    }

    #[test]
    fn test_divide() {
        let mut r = NumberRing::new(400, 100);
        assert_eq!(r.divide(20, 4), Some(5));
        assert_eq!(r.divide(5, 0), None);
    }

    #[test]
    fn test_modulo() {
        let mut r = NumberRing::new(400, 100);
        assert_eq!(r.modulo(17, 5), Some(2));
    }
}
