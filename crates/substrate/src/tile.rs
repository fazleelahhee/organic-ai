use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ToolType {
    Memory,   // store/retrieve signal patterns
    Pattern,  // compare two patterns
    Logic,    // boolean/arithmetic on signals
    Language, // tokenize/detokenize (proto-language)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TileType {
    Empty,
    Resource(f32),
    Hazard,
    Wall,
    Tool(ToolType),
}

impl TileType {
    pub fn is_walkable(&self) -> bool {
        matches!(self, TileType::Empty | TileType::Resource(_) | TileType::Tool(_))
    }
}
