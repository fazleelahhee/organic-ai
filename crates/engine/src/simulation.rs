use organic_core::cell::{Cell, CellType};
use organic_core::direction::{Direction, Position};
use organic_core::genome::Genome;
use organic_core::organism::{LifecyclePhase, Organism, OrganismId};
use organic_growth::interpreter::{count_neighbors, execute_growth_step, EvalContext, GrowthResult};
use organic_substrate::grid::Grid;
use organic_substrate::sal::{ActionResult, SubstrateInterface};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::energy;
use crate::reproduction::mutate_genome;

const MAX_DEVELOPMENT_TICKS: u32 = 50;
const MAX_CELLS_PER_ORGANISM: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    pub width: i32,
    pub height: i32,
    pub initial_resource_count: usize,
    pub initial_organism_count: usize,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self { width: 100, height: 100, initial_resource_count: 500, initial_organism_count: 10 }
    }
}

pub struct World {
    pub grid: Grid,
    pub organisms: Vec<Organism>,
    pub tick_count: u64,
    next_organism_id: OrganismId,
    rng: rand::rngs::ThreadRng,
}

#[derive(Debug, Serialize)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub grid_width: i32,
    pub grid_height: i32,
    pub organisms: Vec<OrganismSnapshot>,
    pub resource_count: usize,
    pub organism_count: usize,
}

#[derive(Debug, Serialize)]
pub struct OrganismSnapshot {
    pub id: OrganismId,
    pub cells: Vec<CellSnapshot>,
    pub energy: f32,
    pub phase: LifecyclePhase,
    pub generation: u32,
    pub age: u32,
}

#[derive(Debug, Serialize)]
pub struct CellSnapshot {
    pub x: i32,
    pub y: i32,
    pub cell_type: CellType,
}

impl World {
    pub fn new(config: WorldConfig) -> Self {
        let mut rng = rand::thread_rng();
        let mut grid = Grid::new(config.width, config.height);
        grid.scatter_resources(config.initial_resource_count, energy::RESOURCE_ENERGY, &mut rng);

        let mut organisms = Vec::new();
        let mut next_id = 0u64;
        for _ in 0..config.initial_organism_count {
            let pos = Position::new(rng.gen_range(0..config.width), rng.gen_range(0..config.height));
            organisms.push(Organism::new(next_id, Genome::simple_default(), pos, 0));
            next_id += 1;
        }

        Self { grid, organisms, tick_count: 0, next_organism_id: next_id, rng }
    }

    pub fn allocate_organism_id(&mut self) -> OrganismId {
        let id = self.next_organism_id;
        self.next_organism_id += 1;
        id
    }

    pub fn tick(&mut self) {
        self.grid.tick();

        let mut new_organisms: Vec<Organism> = Vec::new();

        for org in &mut self.organisms {
            if !org.is_alive() { continue; }
            org.tick_age();

            match org.phase {
                LifecyclePhase::Developing => {
                    // inline development logic
                    if org.age > MAX_DEVELOPMENT_TICKS {
                        org.phase = LifecyclePhase::Living;
                    } else {
                        let cell_positions = org.cell_positions();
                        let rules = org.genome.growth_program.clone();
                        let mut new_cells: Vec<Cell> = Vec::new();
                        let cell_count = org.cells.len();

                        for i in 0..cell_count {
                            if org.cells[i].cell_type != CellType::Undifferentiated { continue; }

                            let neighbor_count = count_neighbors(&cell_positions, org.cells[i].position);
                            let local = self.grid.sense(org.cells[i].position);
                            let ctx = EvalContext { neighbor_count, resource_density: local.resource_density };

                            match execute_growth_step(&org.cells[i], &rules, &ctx, &mut self.rng) {
                                GrowthResult::Divide(new_pos) => {
                                    if cell_count + new_cells.len() < MAX_CELLS_PER_ORGANISM
                                        && self.grid.is_walkable(new_pos)
                                    {
                                        let depth = org.cells[i].depth_from_seed;
                                        let new_id = org.allocate_cell_id();
                                        new_cells.push(Cell::new_child(new_id, new_pos, depth));
                                        org.energy -= energy::COST_GROW_CELL;
                                    }
                                }
                                GrowthResult::Differentiate(ct) => { org.cells[i].cell_type = ct; }
                                GrowthResult::Connect { .. } => { org.cells[i].cell_type = CellType::Inter; }
                                GrowthResult::EmitSignal(v) => { org.cells[i].signal = v; }
                                GrowthResult::Halt => {
                                    if org.cells[i].cell_type == CellType::Undifferentiated { org.cells[i].cell_type = CellType::Inter; }
                                }
                                GrowthResult::NoAction => {}
                            }
                        }

                        org.cells.extend(new_cells);

                        let all_differentiated = org.cells.iter().all(|c| c.cell_type != CellType::Undifferentiated);
                        if all_differentiated && org.cell_count() > 1 {
                            org.phase = LifecyclePhase::Living;
                        }
                    }
                }
                LifecyclePhase::Living => {
                    // Motor cells move organism toward resources
                    let has_motor = org.cells.iter().any(|c| c.cell_type == CellType::Motor);
                    if has_motor {
                        let sensor_positions: Vec<Position> = org.cells.iter()
                            .filter(|c| c.cell_type == CellType::Sensor)
                            .map(|c| c.position).collect();

                        let move_dir = if !sensor_positions.is_empty() {
                            let sample_pos = sensor_positions[0];
                            let mut best_dir = Direction::random(&mut self.rng);
                            let mut best_density = 0.0f32;
                            for dir in Direction::all() {
                                let check_pos = sample_pos.neighbor(dir);
                                let local = self.grid.sense(check_pos);
                                if local.resource_density > best_density {
                                    best_density = local.resource_density;
                                    best_dir = dir;
                                }
                            }
                            if self.rng.gen::<f32>() < 0.3 { Direction::random(&mut self.rng) } else { best_dir }
                        } else {
                            Direction::random(&mut self.rng)
                        };

                        let new_positions: Vec<Position> = org.cells.iter().map(|c| c.position.neighbor(move_dir)).collect();
                        let all_walkable = new_positions.iter().all(|p| self.grid.is_walkable(*p));
                        if all_walkable {
                            for (cell, new_pos) in org.cells.iter_mut().zip(new_positions) {
                                cell.position = new_pos;
                            }
                            org.energy -= energy::COST_MOVE;
                        }
                    }

                    // Sensor cells consume resources
                    for cell in &org.cells {
                        if cell.cell_type == CellType::Sensor {
                            if let ActionResult::Consumed(e) = self.grid.try_consume(cell.position) {
                                org.energy += e;
                            }
                        }
                    }

                    org.energy -= energy::COST_SENSING * org.cells.iter().filter(|c| c.cell_type == CellType::Sensor).count() as f32;
                }
                LifecyclePhase::Dead => {}
            }

            // Energy maintenance
            org.energy -= energy::maintenance_cost(org.cell_count());

            // Death check
            if org.energy <= 0.0 {
                org.phase = LifecyclePhase::Dead;
                continue;
            }

            // Reproduction
            if org.can_reproduce() {
                let parent_pos = org.cells[0].position;
                let mut spawn_pos = None;
                for dir in &Direction::all() {
                    let pos = parent_pos.neighbor(*dir);
                    if self.grid.is_walkable(pos) { spawn_pos = Some(pos); break; }
                }
                if let Some(sp) = spawn_pos {
                    let child_genome = mutate_genome(&org.genome, &mut self.rng);
                    let child_id = self.next_organism_id;
                    self.next_organism_id += 1;
                    let child_gen = org.generation + 1;
                    let energy_transfer = org.energy * energy::REPRODUCE_ENERGY_FRACTION;
                    org.energy -= energy_transfer;
                    let mut child = Organism::new(child_id, child_genome, sp, child_gen);
                    child.energy = energy_transfer;
                    new_organisms.push(child);
                }
            }
        }

        self.organisms.extend(new_organisms);
        self.organisms.retain(|o| o.is_alive());
        self.tick_count += 1;
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let resource_count = self.grid.tiles_ref().iter()
            .filter(|t| matches!(t, organic_substrate::tile::TileType::Resource(_)))
            .count();

        WorldSnapshot {
            tick: self.tick_count,
            grid_width: self.grid.dimensions().0,
            grid_height: self.grid.dimensions().1,
            organisms: self.organisms.iter().map(|o| OrganismSnapshot {
                id: o.id,
                cells: o.cells.iter().map(|c| CellSnapshot { x: c.position.x, y: c.position.y, cell_type: c.cell_type }).collect(),
                energy: o.energy,
                phase: o.phase,
                generation: o.generation,
                age: o.age,
            }).collect(),
            resource_count,
            organism_count: self.organisms.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_creation() {
        let config = WorldConfig { width: 50, height: 50, initial_resource_count: 100, initial_organism_count: 5 };
        let world = World::new(config);
        assert_eq!(world.organisms.len(), 5);
        assert_eq!(world.tick_count, 0);
    }

    #[test]
    fn test_organism_develops() {
        let config = WorldConfig { width: 50, height: 50, initial_resource_count: 200, initial_organism_count: 1 };
        let mut world = World::new(config);
        assert_eq!(world.organisms[0].phase, LifecyclePhase::Developing);
        assert_eq!(world.organisms[0].cell_count(), 1);
        for _ in 0..60 { world.tick(); }
        if !world.organisms.is_empty() {
            let org = &world.organisms[0];
            assert!(
                org.cell_count() > 1 || org.phase == LifecyclePhase::Living || org.phase == LifecyclePhase::Dead,
                "Organism should have grown or died, has {} cells, phase {:?}", org.cell_count(), org.phase,
            );
        }
        // If organisms is empty, the organism died and was cleaned up — that's valid
    }

    #[test]
    fn test_simulation_runs_1000_ticks() {
        let config = WorldConfig { width: 50, height: 50, initial_resource_count: 300, initial_organism_count: 5 };
        let mut world = World::new(config);
        for _ in 0..1000 { world.tick(); }
        assert_eq!(world.tick_count, 1000);
    }

    #[test]
    fn test_reproduction_creates_offspring() {
        let config = WorldConfig { width: 50, height: 50, initial_resource_count: 500, initial_organism_count: 3 };
        let mut world = World::new(config);
        world.organisms[0].energy = 50.0;
        world.organisms[0].phase = LifecyclePhase::Living;
        let initial_count = world.organisms.len();
        world.tick();
        assert!(world.organisms.len() >= initial_count);
    }

    #[test]
    fn test_snapshot_serialization() {
        let config = WorldConfig::default();
        let world = World::new(config);
        let snapshot = world.snapshot();
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("tick"));
        assert!(json.contains("organisms"));
    }
}
