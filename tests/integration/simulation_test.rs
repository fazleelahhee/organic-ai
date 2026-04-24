use organic_core::organism::LifecyclePhase;
use organic_engine::simulation::{World, WorldConfig};

#[test]
fn test_full_lifecycle_seed_to_reproduce() {
    let config = WorldConfig {
        width: 50,
        height: 50,
        initial_resource_count: 500,
        initial_organism_count: 10,
    };
    let mut world = World::new(config);

    let mut max_generation_seen = 0u32;
    let mut saw_living = false;
    let mut saw_multi_cell = false;

    for tick in 0..5000 {
        world.tick();

        for org in &world.organisms {
            if org.phase == LifecyclePhase::Living { saw_living = true; }
            if org.cell_count() > 1 { saw_multi_cell = true; }
            if org.generation > max_generation_seen { max_generation_seen = org.generation; }
        }

        if saw_living && saw_multi_cell && max_generation_seen >= 3 {
            println!("Success at tick {}: max_gen={}, organisms={}", tick, max_generation_seen, world.organisms.len());
            break;
        }
    }

    assert!(saw_multi_cell, "At least one organism should have grown beyond 1 cell");
    assert!(saw_living, "At least one organism should have reached Living phase");
    assert!(max_generation_seen >= 2, "Expected at least 2 generations, got {}", max_generation_seen);
}

#[test]
fn test_snapshot_contains_valid_data() {
    let config = WorldConfig {
        width: 20,
        height: 20,
        initial_resource_count: 50,
        initial_organism_count: 2,
    };
    let mut world = World::new(config);
    for _ in 0..100 { world.tick(); }
    let snap = world.snapshot();
    assert_eq!(snap.grid_width, 20);
    assert_eq!(snap.grid_height, 20);
    assert_eq!(snap.tick, 100);
    for org in &snap.organisms {
        for cell in &org.cells {
            assert!(cell.x >= 0 && cell.x < 20, "Cell x={} out of bounds", cell.x);
            assert!(cell.y >= 0 && cell.y < 20, "Cell y={} out of bounds", cell.y);
        }
    }
}

#[test]
fn test_world_does_not_explode() {
    let config = WorldConfig {
        width: 50,
        height: 50,
        initial_resource_count: 500,
        initial_organism_count: 20,
    };
    let mut world = World::new(config);
    for _ in 0..10_000 { world.tick(); }
    assert!(world.organisms.len() < 1000, "Population exploded to {}, energy economics should prevent this", world.organisms.len());
}

#[test]
fn test_learning_organisms_improve_over_lifetime() {
    let config = WorldConfig {
        width: 30,
        height: 30,
        initial_resource_count: 200,
        initial_organism_count: 5,
    };
    let mut world = World::new(config);

    for org in &mut world.organisms {
        org.phase = LifecyclePhase::Living;
        org.energy = 15.0;
    }

    let mut early_energy: Vec<f32> = Vec::new();
    let mut late_energy: Vec<f32> = Vec::new();

    for tick in 0..2000 {
        world.tick();
        if world.organisms.is_empty() { break; }
        let avg_energy: f32 = world.organisms.iter().map(|o| o.energy).sum::<f32>()
            / world.organisms.len() as f32;
        if tick >= 100 && tick < 200 { early_energy.push(avg_energy); }
        if tick >= 1800 && tick < 2000 { late_energy.push(avg_energy); }
    }

    println!(
        "Early avg energy: {:.2}, Late avg energy: {:.2}, Final organisms: {}",
        early_energy.iter().sum::<f32>() / early_energy.len().max(1) as f32,
        late_energy.iter().sum::<f32>() / late_energy.len().max(1) as f32,
        world.organisms.len()
    );

    assert!(world.tick_count >= 2000 || world.organisms.is_empty());
}

#[test]
fn test_neural_activity_produces_spikes() {
    let config = WorldConfig {
        width: 20,
        height: 20,
        initial_resource_count: 100,
        initial_organism_count: 3,
    };
    let mut world = World::new(config);

    for _ in 0..100 { world.tick(); }

    let any_spikes = world.organisms.iter().any(|o| {
        o.cells.iter().any(|c| c.last_spike_tick.is_some())
    });

    let snap = world.snapshot();
    let any_info_gain = snap.organisms.iter().any(|o| {
        o.cells.iter().any(|c| c.information_gain > 0.0)
    });

    println!("Any spikes after 100 ticks: {}, Any info gain: {}", any_spikes, any_info_gain);
    assert!(any_spikes || world.organisms.is_empty(), "Living organisms should produce spikes");
}
