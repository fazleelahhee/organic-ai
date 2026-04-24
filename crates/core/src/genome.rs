use serde::{Deserialize, Serialize};
use crate::cell::CellType;
use crate::direction::Direction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    AgeGreaterThan(u32),
    NeighborCountLessThan(u32),
    NeighborCountGreaterThan(u32),
    ResourceDensityGreaterThan(f32),
    DepthGreaterThan(u32),
    SignalGreaterThan(f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrowthAction {
    Divide(DirectionChoice),
    Differentiate(CellType),
    Connect { max_distance: u32, target_type: CellType },
    EmitSignal(f32),
    Halt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectionChoice { Fixed(Direction), Random }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthRule { pub conditions: Vec<Condition>, pub action: GrowthAction }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningParams {
    pub stdp_window: u32,
    pub learning_rate: f32,
    pub info_gain_sensitivity: f32,
    pub prediction_depth: u32,
    pub decay_rate: f32,
    pub max_connections: u32,
}

impl Default for LearningParams {
    fn default() -> Self {
        Self { stdp_window: 20, learning_rate: 0.01, info_gain_sensitivity: 0.5, prediction_depth: 1, decay_rate: 0.01, max_connections: 10 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drives { pub curiosity_weight: f32, pub hunger_sensitivity: f32, pub social_drive: f32 }

impl Default for Drives {
    fn default() -> Self { Self { curiosity_weight: 0.5, hunger_sensitivity: 0.7, social_drive: 0.0 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub growth_program: Vec<GrowthRule>,
    pub learning_params: LearningParams,
    pub drives: Drives,
    pub mutation_rate: f32,
}

impl Genome {
    pub fn simple_default() -> Self {
        use Condition::*;
        use GrowthAction::*;
        Self {
            growth_program: vec![
                GrowthRule { conditions: vec![AgeGreaterThan(2), NeighborCountLessThan(4)], action: Divide(DirectionChoice::Random) },
                GrowthRule { conditions: vec![DepthGreaterThan(4), NeighborCountLessThan(3)], action: Differentiate(CellType::Sensor) },
                GrowthRule { conditions: vec![DepthGreaterThan(6)], action: Differentiate(CellType::Motor) },
                GrowthRule { conditions: vec![AgeGreaterThan(8)], action: Connect { max_distance: 3, target_type: CellType::Inter } },
                GrowthRule { conditions: vec![NeighborCountGreaterThan(5)], action: Halt },
            ],
            learning_params: LearningParams::default(),
            drives: Drives::default(),
            mutation_rate: 0.05,
        }
    }
}
