use serde::{Deserialize, Serialize};
use crate::cell::{Cell, CellId};
use crate::direction::Position;
use crate::genome::Genome;

pub type OrganismId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecyclePhase { Developing, Living, Dead }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organism {
    pub id: OrganismId,
    pub genome: Genome,
    pub cells: Vec<Cell>,
    pub energy: f32,
    pub phase: LifecyclePhase,
    pub age: u32,
    pub generation: u32,
    next_cell_id: CellId,
}

impl Organism {
    pub fn new(id: OrganismId, genome: Genome, seed_position: Position, generation: u32) -> Self {
        let seed_cell = Cell::new_seed(0, seed_position);
        Self { id, genome, cells: vec![seed_cell], energy: 10.0, phase: LifecyclePhase::Developing, age: 0, generation, next_cell_id: 1 }
    }

    pub fn allocate_cell_id(&mut self) -> CellId { let id = self.next_cell_id; self.next_cell_id += 1; id }
    pub fn cell_count(&self) -> usize { self.cells.len() }
    pub fn is_alive(&self) -> bool { self.phase != LifecyclePhase::Dead }
    pub fn cell_positions(&self) -> Vec<Position> { self.cells.iter().map(|c| c.position).collect() }
    pub fn can_reproduce(&self) -> bool { self.phase == LifecyclePhase::Living && self.energy > 20.0 }
    pub fn tick_age(&mut self) { self.age += 1; for cell in &mut self.cells { cell.tick_age(); } }
}
