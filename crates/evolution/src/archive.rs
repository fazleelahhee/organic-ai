use organic_core::genome::Genome;
use crate::behavior::BehaviorDescriptor;
use serde::{Deserialize, Serialize};

/// Entry in the QD archive — a genome + its behavior + its fitness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub genome: Genome,
    pub behavior: BehaviorDescriptor,
    pub fitness: f32,
    pub generation: u32,
}

/// MAP-Elites style quality-diversity archive.
/// A 2D grid where each cell holds the best genome for that behavioral niche.
#[derive(Debug, Serialize, Deserialize)]
pub struct QDArchive {
    resolution: usize,
    entries: Vec<Option<ArchiveEntry>>,
    total_insertions: u64,
}

impl QDArchive {
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            entries: vec![None; resolution * resolution],
            total_insertions: 0,
        }
    }

    /// Try to insert a genome into the archive. Returns true if it was accepted
    /// (either empty niche or better fitness than existing).
    pub fn try_insert(&mut self, genome: Genome, behavior: BehaviorDescriptor, fitness: f32, generation: u32) -> bool {
        let (x, y) = behavior.grid_index(self.resolution);
        let idx = y * self.resolution + x;

        let dominated = match &self.entries[idx] {
            None => true,
            Some(existing) => fitness > existing.fitness,
        };

        if dominated {
            self.entries[idx] = Some(ArchiveEntry { genome, behavior, fitness, generation });
            self.total_insertions += 1;
            true
        } else {
            false
        }
    }

    /// How many niches are filled.
    pub fn coverage(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.resolution * self.resolution
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Get the best genomes for seeding new organisms.
    pub fn best_genomes(&self, count: usize) -> Vec<&Genome> {
        let mut filled: Vec<&ArchiveEntry> = self.entries.iter().filter_map(|e| e.as_ref()).collect();
        filled.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        filled.iter().take(count).map(|e| &e.genome).collect()
    }

    /// Compute novelty score for a behavior (avg distance to k-nearest in archive).
    pub fn novelty_score(&self, behavior: &BehaviorDescriptor, k: usize) -> f32 {
        let mut distances: Vec<f32> = self.entries.iter()
            .filter_map(|e| e.as_ref())
            .map(|e| behavior.distance(&e.behavior))
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = k.min(distances.len());
        if k == 0 { return 1.0; }
        distances[..k].iter().sum::<f32>() / k as f32
    }

    /// Get a snapshot of the archive for visualization.
    pub fn snapshot(&self) -> ArchiveSnapshot {
        let mut cells = Vec::new();
        for y in 0..self.resolution {
            for x in 0..self.resolution {
                let idx = y * self.resolution + x;
                if let Some(entry) = &self.entries[idx] {
                    cells.push(ArchiveCellSnapshot {
                        x, y,
                        fitness: entry.fitness,
                        generation: entry.generation,
                    });
                }
            }
        }
        ArchiveSnapshot {
            resolution: self.resolution,
            coverage: self.coverage(),
            total_insertions: self.total_insertions,
            cells,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ArchiveSnapshot {
    pub resolution: usize,
    pub coverage: usize,
    pub total_insertions: u64,
    pub cells: Vec<ArchiveCellSnapshot>,
}

#[derive(Debug, Serialize)]
pub struct ArchiveCellSnapshot {
    pub x: usize,
    pub y: usize,
    pub fitness: f32,
    pub generation: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use organic_core::genome::Genome;

    fn make_behavior(sr: f32, mr: f32) -> BehaviorDescriptor {
        BehaviorDescriptor {
            cell_count: 0.5, sensor_ratio: sr, motor_ratio: mr,
            avg_info_gain: 0.1, movement_rate: 0.5, connection_density: 0.3,
        }
    }

    #[test]
    fn test_empty_archive() {
        let archive = QDArchive::new(10);
        assert_eq!(archive.coverage(), 0);
        assert_eq!(archive.capacity(), 100);
    }

    #[test]
    fn test_insert_into_empty_niche() {
        let mut archive = QDArchive::new(10);
        let inserted = archive.try_insert(Genome::simple_default(), make_behavior(0.5, 0.3), 10.0, 0);
        assert!(inserted);
        assert_eq!(archive.coverage(), 1);
    }

    #[test]
    fn test_better_fitness_replaces() {
        let mut archive = QDArchive::new(10);
        archive.try_insert(Genome::simple_default(), make_behavior(0.5, 0.3), 10.0, 0);
        let replaced = archive.try_insert(Genome::simple_default(), make_behavior(0.5, 0.3), 20.0, 1);
        assert!(replaced);
        assert_eq!(archive.coverage(), 1); // same niche
    }

    #[test]
    fn test_worse_fitness_rejected() {
        let mut archive = QDArchive::new(10);
        archive.try_insert(Genome::simple_default(), make_behavior(0.5, 0.3), 20.0, 0);
        let rejected = archive.try_insert(Genome::simple_default(), make_behavior(0.5, 0.3), 10.0, 1);
        assert!(!rejected);
    }

    #[test]
    fn test_different_niches_both_kept() {
        let mut archive = QDArchive::new(10);
        archive.try_insert(Genome::simple_default(), make_behavior(0.1, 0.1), 10.0, 0);
        archive.try_insert(Genome::simple_default(), make_behavior(0.9, 0.9), 10.0, 0);
        assert_eq!(archive.coverage(), 2);
    }

    #[test]
    fn test_novelty_score() {
        let mut archive = QDArchive::new(10);
        archive.try_insert(Genome::simple_default(), make_behavior(0.1, 0.1), 10.0, 0);
        archive.try_insert(Genome::simple_default(), make_behavior(0.2, 0.2), 10.0, 0);

        let novel = archive.novelty_score(&make_behavior(0.9, 0.9), 2);
        let familiar = archive.novelty_score(&make_behavior(0.15, 0.15), 2);
        assert!(novel > familiar, "Novel behavior should score higher");
    }

    #[test]
    fn test_archive_snapshot() {
        let mut archive = QDArchive::new(10);
        archive.try_insert(Genome::simple_default(), make_behavior(0.5, 0.3), 10.0, 1);
        let snap = archive.snapshot();
        assert_eq!(snap.coverage, 1);
        assert_eq!(snap.cells.len(), 1);
    }
}
