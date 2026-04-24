pub const COST_EXIST_PER_CELL: f32 = 0.01;
pub const COST_SENSING: f32 = 0.005;
pub const COST_MOVE: f32 = 0.1;
pub const COST_GROW_CELL: f32 = 0.5;
pub const REPRODUCE_ENERGY_FRACTION: f32 = 0.5;
pub const REPRODUCE_THRESHOLD: f32 = 20.0;
pub const RESOURCE_ENERGY: f32 = 5.0;

pub fn maintenance_cost(cell_count: usize) -> f32 {
    cell_count as f32 * COST_EXIST_PER_CELL
}
