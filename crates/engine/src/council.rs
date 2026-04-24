use organic_core::organism::{Organism, OrganismId};
use serde::{Deserialize, Serialize};

/// The organism council — tracks the most capable organisms.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Council {
    pub member_ids: Vec<OrganismId>,
    pub max_size: usize,
}

impl Council {
    pub fn new(max_size: usize) -> Self {
        Self { member_ids: Vec::new(), max_size }
    }

    /// Update council membership based on organism fitness (energy + age).
    pub fn update(&mut self, organisms: &[Organism]) {
        let mut candidates: Vec<(OrganismId, f32)> = organisms.iter()
            .filter(|o| o.is_alive())
            .map(|o| {
                let fitness = o.energy + (o.age as f32 * 0.1) + (o.generation as f32 * 0.5);
                (o.id, fitness)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.member_ids = candidates.iter().take(self.max_size).map(|(id, _)| *id).collect();
    }

    pub fn is_member(&self, id: OrganismId) -> bool {
        self.member_ids.contains(&id)
    }

    pub fn size(&self) -> usize {
        self.member_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use organic_core::direction::Position;
    use organic_core::genome::Genome;
    use organic_core::organism::Organism;

    #[test]
    fn test_council_selects_best() {
        let mut orgs = vec![
            Organism::new(0, Genome::simple_default(), Position::new(0, 0), 0),
            Organism::new(1, Genome::simple_default(), Position::new(1, 1), 0),
            Organism::new(2, Genome::simple_default(), Position::new(2, 2), 0),
        ];
        orgs[0].energy = 10.0;
        orgs[1].energy = 50.0;
        orgs[2].energy = 30.0;

        let mut council = Council::new(2);
        council.update(&orgs);
        assert_eq!(council.size(), 2);
        assert!(council.is_member(1)); // highest energy
        assert!(council.is_member(2)); // second highest
        assert!(!council.is_member(0));
    }
}
