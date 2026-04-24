use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifParams {
    pub threshold: f32,
    pub leak_rate: f32,
    pub reset_potential: f32,
}

impl Default for LifParams {
    fn default() -> Self {
        Self { threshold: 1.0, leak_rate: 0.05, reset_potential: 0.0 }
    }
}

pub fn integrate_and_fire(potential: &mut f32, input: f32, params: &LifParams) -> bool {
    *potential -= params.leak_rate;
    if *potential < 0.0 { *potential = 0.0; }
    *potential += input;
    if *potential >= params.threshold {
        *potential = params.reset_potential;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_threshold_no_spike() {
        let mut potential = 0.3;
        let fired = integrate_and_fire(&mut potential, 0.2, &LifParams::default());
        assert!(!fired);
        assert!(potential > 0.0);
    }

    #[test]
    fn test_above_threshold_spikes_and_resets() {
        let mut potential = 0.9;
        let fired = integrate_and_fire(&mut potential, 0.5, &LifParams::default());
        assert!(fired);
        assert_eq!(potential, 0.0);
    }

    #[test]
    fn test_leak_reduces_potential() {
        let mut potential = 0.5;
        let _ = integrate_and_fire(&mut potential, 0.0, &LifParams::default());
        assert!(potential < 0.5);
    }

    #[test]
    fn test_potential_does_not_go_negative() {
        let mut potential = 0.01;
        let _ = integrate_and_fire(&mut potential, 0.0, &LifParams::default());
        assert!(potential >= 0.0);
    }
}
