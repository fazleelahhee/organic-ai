use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdpParams {
    pub window: u64,
    pub a_plus: f32,
    pub a_minus: f32,
    pub weight_max: f32,
    pub weight_min: f32,
}

impl Default for StdpParams {
    fn default() -> Self {
        Self { window: 20, a_plus: 0.02, a_minus: 0.01, weight_max: 1.0, weight_min: 0.0 }
    }
}

pub fn stdp_weight_change(pre_tick: u64, post_tick: u64, params: &StdpParams) -> f32 {
    if pre_tick == post_tick { return 0.0; }
    let dt = post_tick as i64 - pre_tick as i64;
    let abs_dt = dt.unsigned_abs();
    if abs_dt > params.window { return 0.0; }
    let decay = 1.0 - (abs_dt as f32 / params.window as f32);
    if dt > 0 { params.a_plus * decay } else { -params.a_minus * decay }
}

pub fn apply_stdp(weight: &mut f32, pre_tick: u64, post_tick: u64, info_gain: f32, params: &StdpParams) {
    let base_change = stdp_weight_change(pre_tick, post_tick, params);
    let modulated = base_change * (1.0 + info_gain);
    *weight = (*weight + modulated).clamp(params.weight_min, params.weight_max);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_before_post_strengthens() {
        let delta = stdp_weight_change(10, 12, &StdpParams::default());
        assert!(delta > 0.0);
    }

    #[test]
    fn test_post_before_pre_weakens() {
        let delta = stdp_weight_change(12, 10, &StdpParams::default());
        assert!(delta < 0.0);
    }

    #[test]
    fn test_simultaneous_no_change() {
        let delta = stdp_weight_change(10, 10, &StdpParams::default());
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_outside_window_no_change() {
        let params = StdpParams { window: 20, ..StdpParams::default() };
        let delta = stdp_weight_change(10, 50, &params);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_apply_stdp_clamps_weight() {
        let mut weight = 0.95;
        apply_stdp(&mut weight, 10, 12, 1.0, &StdpParams::default());
        assert!(weight <= 1.0);
        let mut weight2 = 0.05;
        apply_stdp(&mut weight2, 12, 10, 1.0, &StdpParams::default());
        assert!(weight2 >= 0.0);
    }
}
