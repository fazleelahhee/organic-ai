use serde::{Deserialize, Serialize};
use crate::direction::Position;

pub type CellId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType { Sensor, Inter, Motor, Reproductive, Undifferentiated }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse { pub target: CellId, pub weight: f32 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub id: CellId,
    pub cell_type: CellType,
    pub position: Position,
    pub age: u32,
    pub depth_from_seed: u32,
    pub signal: f32,
    pub connections: Vec<Synapse>,
}

impl Cell {
    pub fn new_seed(id: CellId, position: Position) -> Self {
        Self { id, cell_type: CellType::Undifferentiated, position, age: 0, depth_from_seed: 0, signal: 0.0, connections: Vec::new() }
    }

    pub fn new_child(id: CellId, position: Position, parent_depth: u32) -> Self {
        Self { id, cell_type: CellType::Undifferentiated, position, age: 0, depth_from_seed: parent_depth + 1, signal: 0.0, connections: Vec::new() }
    }

    pub fn tick_age(&mut self) { self.age += 1; }
}
