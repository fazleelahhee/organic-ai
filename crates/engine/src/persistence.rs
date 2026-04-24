use crate::simulation::World;
use std::fs;
use std::path::Path;

/// Save world state to a binary file using bincode.
pub fn save_world(world: &World, path: &str) -> Result<(), String> {
    let data = bincode::serialize(world).map_err(|e| format!("Serialize error: {}", e))?;
    fs::write(path, &data).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}

/// Load world state from a binary file.
pub fn load_world(path: &str) -> Result<World, String> {
    if !Path::new(path).exists() {
        return Err("Save file not found".to_string());
    }
    let data = fs::read(path).map_err(|e| format!("Read error: {}", e))?;
    let world: World = bincode::deserialize(&data).map_err(|e| format!("Deserialize error: {}", e))?;
    Ok(world)
}

/// Auto-save path for the default world.
pub fn default_save_path() -> String {
    "data/world_save.bin".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::WorldConfig;

    #[test]
    fn test_save_and_load_roundtrip() {
        let config = WorldConfig { width: 20, height: 20, initial_resource_count: 50, initial_organism_count: 3 };
        let mut world = World::new(config);
        for _ in 0..100 { world.tick(); }

        let path = "/tmp/organic_ai_test_save.bin";
        save_world(&world, path).unwrap();
        let loaded = load_world(path).unwrap();

        assert_eq!(loaded.tick_count, world.tick_count);
        assert_eq!(loaded.organisms.len(), world.organisms.len());

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_nonexistent() {
        let result = load_world("/tmp/nonexistent_organic_save.bin");
        assert!(result.is_err());
    }
}
