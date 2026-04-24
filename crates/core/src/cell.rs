use serde::{Deserialize, Serialize};
use crate::direction::Position;

pub type CellId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType { Sensor, Inter, Motor, Reproductive, Undifferentiated }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub target: CellId,
    pub weight: f32,
    pub last_pre_tick: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub id: CellId,
    pub cell_type: CellType,
    pub position: Position,
    pub age: u32,
    pub depth_from_seed: u32,
    pub signal: f32,
    pub connections: Vec<Synapse>,
    pub membrane_potential: f32,
    pub spike_active: bool,
    pub last_spike_tick: Option<u64>,
    pub predicted_input: f32,
    pub information_gain: f32,
}

impl Cell {
    pub fn new_seed(id: CellId, position: Position) -> Self {
        Self { id, cell_type: CellType::Undifferentiated, position, age: 0, depth_from_seed: 0, signal: 0.0, connections: Vec::new(), membrane_potential: 0.0, spike_active: false, last_spike_tick: None, predicted_input: 0.0, information_gain: 0.0 }
    }

    pub fn new_child(id: CellId, position: Position, parent_depth: u32) -> Self {
        Self { id, cell_type: CellType::Undifferentiated, position, age: 0, depth_from_seed: parent_depth + 1, signal: 0.0, connections: Vec::new(), membrane_potential: 0.0, spike_active: false, last_spike_tick: None, predicted_input: 0.0, information_gain: 0.0 }
    }

    pub fn tick_age(&mut self) { self.age += 1; }
}
