use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TileType {
    Empty,
    Resource(f32),
    Hazard,
    Wall,
}

impl TileType {
    pub fn is_walkable(&self) -> bool {
        matches!(self, TileType::Empty | TileType::Resource(_))
    }
}
