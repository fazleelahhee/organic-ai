/// Apply a basic operation on two signal values.
#[derive(Debug, Clone, Copy)]
pub enum LogicOp { And, Or, Xor, Add, Multiply }

pub fn apply_logic(op: LogicOp, a: f32, b: f32) -> f32 {
    match op {
        LogicOp::And => if a > 0.5 && b > 0.5 { 1.0 } else { 0.0 },
        LogicOp::Or => if a > 0.5 || b > 0.5 { 1.0 } else { 0.0 },
        LogicOp::Xor => if (a > 0.5) != (b > 0.5) { 1.0 } else { 0.0 },
        LogicOp::Add => (a + b).clamp(0.0, 1.0),
        LogicOp::Multiply => a * b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and() {
        assert_eq!(apply_logic(LogicOp::And, 0.8, 0.9), 1.0);
        assert_eq!(apply_logic(LogicOp::And, 0.3, 0.9), 0.0);
    }

    #[test]
    fn test_xor() {
        assert_eq!(apply_logic(LogicOp::Xor, 0.8, 0.3), 1.0);
        assert_eq!(apply_logic(LogicOp::Xor, 0.8, 0.9), 0.0);
    }

    #[test]
    fn test_add_clamps() {
        assert_eq!(apply_logic(LogicOp::Add, 0.8, 0.5), 1.0);
    }
}
