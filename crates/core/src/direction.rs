use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }

    pub fn neighbor(self, dir: Direction) -> Self {
        match dir {
            Direction::North => Self { x: self.x, y: self.y - 1 },
            Direction::South => Self { x: self.x, y: self.y + 1 },
            Direction::East => Self { x: self.x + 1, y: self.y },
            Direction::West => Self { x: self.x - 1, y: self.y },
        }
    }

    pub fn neighbors(self) -> [Position; 4] {
        [
            self.neighbor(Direction::North),
            self.neighbor(Direction::South),
            self.neighbor(Direction::East),
            self.neighbor(Direction::West),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction { North, South, East, West }

impl Direction {
    pub fn all() -> [Direction; 4] {
        [Direction::North, Direction::South, Direction::East, Direction::West]
    }

    pub fn random(rng: &mut impl rand::Rng) -> Self {
        use rand::seq::SliceRandom;
        *Self::all().choose(rng).unwrap()
    }
}
