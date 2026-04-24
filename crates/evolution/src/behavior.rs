use organic_core::cell::CellType;
use organic_core::organism::Organism;
use serde::{Deserialize, Serialize};

/// Behavioral descriptor — characterizes HOW an organism behaves, not how well.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorDescriptor {
    pub cell_count: f32,         // normalized 0-1
    pub sensor_ratio: f32,       // fraction of cells that are sensors
    pub motor_ratio: f32,        // fraction that are motors
    pub avg_info_gain: f32,      // average curiosity/surprise
    pub movement_rate: f32,      // how much it moves (energy spent on movement)
    pub connection_density: f32, // avg connections per cell
}

impl BehaviorDescriptor {
    pub fn from_organism(org: &Organism) -> Self {
        let n = org.cells.len().max(1) as f32;
        let sensors = org.cells.iter().filter(|c| c.cell_type == CellType::Sensor).count() as f32;
        let motors = org.cells.iter().filter(|c| c.cell_type == CellType::Motor).count() as f32;
        let avg_info = org.cells.iter().map(|c| c.information_gain).sum::<f32>() / n;
        let avg_conn = org.cells.iter().map(|c| c.connections.len() as f32).sum::<f32>() / n;

        Self {
            cell_count: (n / 30.0).clamp(0.0, 1.0),
            sensor_ratio: sensors / n,
            motor_ratio: motors / n,
            avg_info_gain: avg_info,
            movement_rate: 0.0, // updated externally
            connection_density: (avg_conn / 10.0).clamp(0.0, 1.0),
        }
    }

    /// Distance between two behavior descriptors (for novelty).
    pub fn distance(&self, other: &BehaviorDescriptor) -> f32 {
        let d = [
            self.cell_count - other.cell_count,
            self.sensor_ratio - other.sensor_ratio,
            self.motor_ratio - other.motor_ratio,
            self.avg_info_gain - other.avg_info_gain,
            self.movement_rate - other.movement_rate,
            self.connection_density - other.connection_density,
        ];
        d.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Discretize to grid cell for MAP-Elites style archive.
    /// Returns (x, y) indices into a 2D behavioral space.
    pub fn grid_index(&self, resolution: usize) -> (usize, usize) {
        let x = ((self.sensor_ratio) * resolution as f32).min(resolution as f32 - 1.0).max(0.0) as usize;
        let y = ((self.motor_ratio) * resolution as f32).min(resolution as f32 - 1.0).max(0.0) as usize;
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use organic_core::cell::CellType;
    use organic_core::direction::Position;
    use organic_core::genome::Genome;
    use organic_core::organism::Organism;

    #[test]
    fn test_behavior_from_organism() {
        let mut org = Organism::new(0, Genome::simple_default(), Position::new(5, 5), 0);
        org.cells[0].cell_type = CellType::Sensor;
        let bd = BehaviorDescriptor::from_organism(&org);
        assert_eq!(bd.sensor_ratio, 1.0);
        assert_eq!(bd.motor_ratio, 0.0);
    }

    #[test]
    fn test_distance_same_is_zero() {
        let bd = BehaviorDescriptor {
            cell_count: 0.5, sensor_ratio: 0.3, motor_ratio: 0.2,
            avg_info_gain: 0.1, movement_rate: 0.5, connection_density: 0.4,
        };
        assert_eq!(bd.distance(&bd), 0.0);
    }

    #[test]
    fn test_grid_index_in_bounds() {
        let bd = BehaviorDescriptor {
            cell_count: 0.5, sensor_ratio: 0.8, motor_ratio: 0.3,
            avg_info_gain: 0.1, movement_rate: 0.5, connection_density: 0.4,
        };
        let (x, y) = bd.grid_index(10);
        assert!(x < 10);
        assert!(y < 10);
    }
}
