use organic_core::genome::{Condition, Genome};
use rand::Rng;

pub fn mutate_genome(genome: &Genome, rng: &mut impl Rng) -> Genome {
    let mut new = genome.clone();
    let rate = genome.mutation_rate;

    // Mutate learning params
    if rng.gen::<f32>() < rate {
        new.learning_params.learning_rate =
            (new.learning_params.learning_rate + rng.gen_range(-0.005..0.005)).clamp(0.001, 0.1);
    }
    if rng.gen::<f32>() < rate {
        new.learning_params.stdp_window =
            (new.learning_params.stdp_window as i32 + rng.gen_range(-2..=2)).clamp(5, 50) as u32;
    }

    // Mutate drives
    if rng.gen::<f32>() < rate {
        new.drives.curiosity_weight =
            (new.drives.curiosity_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
    }
    if rng.gen::<f32>() < rate {
        new.drives.hunger_sensitivity =
            (new.drives.hunger_sensitivity + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
    }

    // Mutate growth rules — point mutation on numeric values
    for rule in &mut new.growth_program {
        for cond in &mut rule.conditions {
            if rng.gen::<f32>() < rate {
                match cond {
                    Condition::AgeGreaterThan(ref mut n) => {
                        *n = (*n as i32 + rng.gen_range(-1..=1)).max(0) as u32;
                    }
                    Condition::NeighborCountLessThan(ref mut n)
                    | Condition::NeighborCountGreaterThan(ref mut n) => {
                        *n = (*n as i32 + rng.gen_range(-1..=1)).max(0) as u32;
                    }
                    Condition::DepthGreaterThan(ref mut n) => {
                        *n = (*n as i32 + rng.gen_range(-1..=1)).max(0) as u32;
                    }
                    Condition::ResourceDensityGreaterThan(ref mut v)
                    | Condition::SignalGreaterThan(ref mut v) => {
                        *v = (*v + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    // Rule deletion (rare)
    if rng.gen::<f32>() < rate * 0.3 && new.growth_program.len() > 2 {
        let idx = rng.gen_range(0..new.growth_program.len());
        new.growth_program.remove(idx);
    }

    // Rule swap (rare)
    if rng.gen::<f32>() < rate * 0.3 && new.growth_program.len() >= 2 {
        let a = rng.gen_range(0..new.growth_program.len());
        let b = rng.gen_range(0..new.growth_program.len());
        new.growth_program.swap(a, b);
    }

    // Mutate mutation rate itself
    if rng.gen::<f32>() < 0.1 {
        new.mutation_rate = (new.mutation_rate + rng.gen_range(-0.01..0.01)).clamp(0.01, 0.2);
    }

    new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutate_genome_returns_different_genome() {
        let genome = Genome::simple_default();
        let mut rng = rand::thread_rng();
        let mut found_difference = false;
        for _ in 0..100 {
            let mutated = mutate_genome(&genome, &mut rng);
            if mutated.learning_params.learning_rate != genome.learning_params.learning_rate
                || mutated.drives.curiosity_weight != genome.drives.curiosity_weight
                || mutated.growth_program.len() != genome.growth_program.len()
            {
                found_difference = true;
                break;
            }
        }
        assert!(found_difference, "Mutation should produce differences after 100 attempts");
    }

    #[test]
    fn test_mutated_values_stay_in_range() {
        let genome = Genome::simple_default();
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mutated = mutate_genome(&genome, &mut rng);
            assert!(mutated.learning_params.learning_rate >= 0.001);
            assert!(mutated.learning_params.learning_rate <= 0.1);
            assert!(mutated.drives.curiosity_weight >= 0.0);
            assert!(mutated.drives.curiosity_weight <= 1.0);
            assert!(mutated.mutation_rate >= 0.01);
            assert!(mutated.mutation_rate <= 0.2);
        }
    }
}
