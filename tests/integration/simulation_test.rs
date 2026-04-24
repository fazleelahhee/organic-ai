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
