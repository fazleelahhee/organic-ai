pub fn compute_information_gain(predicted: f32, actual: f32) -> f32 {
    (actual - predicted).abs().clamp(0.0, 1.0)
}

pub fn update_prediction(predicted: &mut f32, actual: f32, learning_rate: f32) {
    *predicted += learning_rate * (actual - *predicted);
}

pub fn homeostatic_curiosity_drive(current_info_gain: f32, target_info_gain: f32) -> f32 {
    target_info_gain - current_info_gain
}

pub fn substrate_to_signal(resource_density: f32) -> f32 {
    (resource_density / 5.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_error_is_surprise() {
        let gain = compute_information_gain(0.3, 0.8);
        assert!(gain > 0.0);
        assert!(gain <= 1.0);
    }

    #[test]
    fn test_no_surprise_when_correct() {
        let gain = compute_information_gain(0.5, 0.5);
        assert_eq!(gain, 0.0);
    }

    #[test]
    fn test_update_prediction_moves_toward_actual() {
        let mut predicted = 0.2;
        update_prediction(&mut predicted, 0.8, 0.1);
        assert!(predicted > 0.2);
        assert!(predicted < 0.8);
    }

    #[test]
    fn test_homeostatic_drive() {
        assert!(homeostatic_curiosity_drive(0.1, 0.5) > 0.0);
        assert!(homeostatic_curiosity_drive(0.9, 0.5) < 0.0);
        assert_eq!(homeostatic_curiosity_drive(0.5, 0.5), 0.0);
    }
}
