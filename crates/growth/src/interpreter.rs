use organic_core::cell::{Cell, CellId, CellType};
use organic_core::direction::{Direction, Position};
use organic_core::genome::{Condition, DirectionChoice, GrowthAction, GrowthRule};
use organic_substrate::sal::SubstrateInterface;
use rand::Rng;

pub struct EvalContext {
    pub neighbor_count: u32,
    pub resource_density: f32,
}

pub fn evaluate_condition(cond: &Condition, cell: &Cell, ctx: &EvalContext) -> bool {
    match cond {
        Condition::AgeGreaterThan(n) => cell.age > *n,
        Condition::NeighborCountLessThan(n) => ctx.neighbor_count < *n,
        Condition::NeighborCountGreaterThan(n) => ctx.neighbor_count > *n,
        Condition::ResourceDensityGreaterThan(v) => ctx.resource_density > *v,
        Condition::DepthGreaterThan(n) => cell.depth_from_seed > *n,
        Condition::SignalGreaterThan(v) => cell.signal > *v,
    }
}

pub fn evaluate_rule(rule: &GrowthRule, cell: &Cell, ctx: &EvalContext) -> bool {
    rule.conditions.iter().all(|c| evaluate_condition(c, cell, ctx))
}

pub enum GrowthResult {
    Divide(Position),
    Differentiate(CellType),
    Connect { max_distance: u32, target_type: CellType },
    EmitSignal(f32),
    Halt,
    NoAction,
}

pub fn execute_growth_step(
    cell: &Cell,
    rules: &[GrowthRule],
    ctx: &EvalContext,
    rng: &mut impl Rng,
) -> GrowthResult {
    for rule in rules {
        if evaluate_rule(rule, cell, ctx) {
            return match &rule.action {
                GrowthAction::Divide(dir_choice) => {
                    let dir = match dir_choice {
                        DirectionChoice::Fixed(d) => *d,
                        DirectionChoice::Random => Direction::random(rng),
                    };
                    GrowthResult::Divide(cell.position.neighbor(dir))
                }
                GrowthAction::Differentiate(ct) => GrowthResult::Differentiate(*ct),
                GrowthAction::Connect { max_distance, target_type } => {
                    GrowthResult::Connect { max_distance: *max_distance, target_type: *target_type }
                }
                GrowthAction::EmitSignal(v) => GrowthResult::EmitSignal(*v),
                GrowthAction::Halt => GrowthResult::Halt,
            };
        }
    }
    GrowthResult::NoAction
}

pub fn count_neighbors(cell_positions: &[Position], pos: Position) -> u32 {
    let neighbors = pos.neighbors();
    cell_positions.iter().filter(|p| neighbors.contains(p) && **p != pos).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use organic_core::cell::Cell;
    use organic_core::direction::Position;
    use organic_core::genome::Condition;

    #[test]
    fn test_age_condition() {
        let mut cell = Cell::new_seed(0, Position::new(5, 5));
        cell.age = 5;
        let ctx = EvalContext { neighbor_count: 2, resource_density: 0.5 };
        assert!(evaluate_condition(&Condition::AgeGreaterThan(3), &cell, &ctx));
        assert!(!evaluate_condition(&Condition::AgeGreaterThan(10), &cell, &ctx));
    }

    #[test]
    fn test_neighbor_count_condition() {
        let cell = Cell::new_seed(0, Position::new(5, 5));
        let ctx = EvalContext { neighbor_count: 2, resource_density: 0.5 };
        assert!(evaluate_condition(&Condition::NeighborCountLessThan(4), &cell, &ctx));
        assert!(!evaluate_condition(&Condition::NeighborCountLessThan(1), &cell, &ctx));
    }

    #[test]
    fn test_rule_evaluation_all_conditions_must_match() {
        let mut cell = Cell::new_seed(0, Position::new(5, 5));
        cell.age = 5;
        let ctx = EvalContext { neighbor_count: 2, resource_density: 0.5 };

        let rule = GrowthRule {
            conditions: vec![Condition::AgeGreaterThan(3), Condition::NeighborCountLessThan(4)],
            action: GrowthAction::Divide(DirectionChoice::Random),
        };
        assert!(evaluate_rule(&rule, &cell, &ctx));

        let rule2 = GrowthRule {
            conditions: vec![Condition::AgeGreaterThan(10), Condition::NeighborCountLessThan(4)],
            action: GrowthAction::Halt,
        };
        assert!(!evaluate_rule(&rule2, &cell, &ctx));
    }

    #[test]
    fn test_first_matching_rule_fires() {
        let mut cell = Cell::new_seed(0, Position::new(5, 5));
        cell.age = 5;
        let ctx = EvalContext { neighbor_count: 2, resource_density: 0.5 };

        let rules = vec![
            GrowthRule {
                conditions: vec![Condition::AgeGreaterThan(10)],
                action: GrowthAction::Halt,
            },
            GrowthRule {
                conditions: vec![Condition::AgeGreaterThan(3)],
                action: GrowthAction::Differentiate(CellType::Sensor),
            },
        ];

        let mut rng = rand::thread_rng();
        match execute_growth_step(&cell, &rules, &ctx, &mut rng) {
            GrowthResult::Differentiate(CellType::Sensor) => {}
            _ => panic!("Expected Differentiate(Sensor)"),
        }
    }

    #[test]
    fn test_count_neighbors() {
        let positions = vec![
            Position::new(5, 5),
            Position::new(5, 4),
            Position::new(6, 5),
            Position::new(10, 10),
        ];
        assert_eq!(count_neighbors(&positions, Position::new(5, 5)), 2);
        assert_eq!(count_neighbors(&positions, Position::new(10, 10)), 0);
    }
}
