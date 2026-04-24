/// Compute similarity between two signal patterns (cosine similarity).
/// Returns 0.0 for completely different, 1.0 for identical.
pub fn pattern_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let len = a.len().min(b.len());
    let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
    (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_patterns() {
        assert!((pattern_similarity(&[1.0, 2.0], &[1.0, 2.0]) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_orthogonal_patterns() {
        assert!(pattern_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.01);
    }

    #[test]
    fn test_empty_patterns() {
        assert_eq!(pattern_similarity(&[], &[1.0]), 0.0);
    }
}
