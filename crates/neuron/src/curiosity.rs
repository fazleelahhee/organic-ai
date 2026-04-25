pub fn compute_information_gain(predicted: f32, actual: f32) -> f32 {
    (actual - predicted).abs().clamp(0.0, 1.0)
}

pub fn update_prediction(predicted: &mut f32, actual: f32, learning_rate: f32) {
    *predicted += learning_rate * (actual - *predicted);
}

pub fn homeostatic_curiosity_drive(current_info_gain: f32, target_info_gain: f32) -> f32 {
    target_info_gain - current_info_gain
}

/// Prediction error from HDC similarity: how far the predicted response was
/// from the actual one.  1.0 = maximally wrong, 0.0 = perfect prediction.
pub fn compute_hdc_prediction_error(predicted_similarity: f32) -> f32 {
    (1.0 - predicted_similarity).clamp(0.0, 1.0)
}

/// Weighted average of HDC prediction error and predictive-coding error.
pub fn compute_combined_error(hdc_error: f32, coding_error: f32, hdc_weight: f32) -> f32 {
    let w = hdc_weight.clamp(0.0, 1.0);
    w * hdc_error + (1.0 - w) * coding_error
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
    fn test_hdc_prediction_error() {
        assert_eq!(compute_hdc_prediction_error(1.0), 0.0);
        assert_eq!(compute_hdc_prediction_error(0.0), 1.0);
        let mid = compute_hdc_prediction_error(0.6);
        assert!((mid - 0.4).abs() < 1e-6);
        // Clamp negative similarity
        assert_eq!(compute_hdc_prediction_error(1.5), 0.0);
    }

    #[test]
    fn test_combined_error() {
        // Pure HDC weight
        let e = compute_combined_error(0.8, 0.2, 1.0);
        assert!((e - 0.8).abs() < 1e-6);
        // Pure coding weight
        let e2 = compute_combined_error(0.8, 0.2, 0.0);
        assert!((e2 - 0.2).abs() < 1e-6);
        // 50/50
        let e3 = compute_combined_error(0.8, 0.2, 0.5);
        assert!((e3 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_homeostatic_drive() {
        assert!(homeostatic_curiosity_drive(0.1, 0.5) > 0.0);
        assert!(homeostatic_curiosity_drive(0.9, 0.5) < 0.0);
        assert_eq!(homeostatic_curiosity_drive(0.5, 0.5), 0.0);
    }
}
