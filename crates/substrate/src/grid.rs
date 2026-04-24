use organic_core::direction::{Direction, Position};
use crate::sal::{ActionResult, LocalState, SubstrateInterface};
use crate::tile::TileType;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    width: i32,
    height: i32,
    tiles: Vec<TileType>,
    resource_regen_rate: f32,
    resource_max: f32,
}

impl Grid {
    pub fn new(width: i32, height: i32) -> Self {
        let size = (width * height) as usize;
        Self { width, height, tiles: vec![TileType::Empty; size], resource_regen_rate: 0.01, resource_max: 5.0 }
    }

    pub fn scatter_resources(&mut self, count: usize, energy: f32, rng: &mut impl Rng) {
        for _ in 0..count {
            let x = rng.gen_range(0..self.width);
            let y = rng.gen_range(0..self.height);
            if let Some(idx) = self.index(Position::new(x, y)) {
                self.tiles[idx] = TileType::Resource(energy);
            }
        }
    }

    fn index(&self, pos: Position) -> Option<usize> {
        if pos.x >= 0 && pos.x < self.width && pos.y >= 0 && pos.y < self.height {
            Some((pos.y * self.width + pos.x) as usize)
        } else { None }
    }

    pub fn get_tile(&self, pos: Position) -> TileType {
        self.index(pos).map(|i| self.tiles[i]).unwrap_or(TileType::Wall)
    }

    fn set_tile(&mut self, pos: Position, tile: TileType) {
        if let Some(i) = self.index(pos) { self.tiles[i] = tile; }
    }

    pub fn set_tile_pub(&mut self, pos: Position, tile: TileType) {
        if let Some(i) = self.index(pos) { self.tiles[i] = tile; }
    }

    fn resource_density_at(&self, pos: Position) -> f32 {
        let neighbors = pos.neighbors();
        let mut total = 0.0;
        let mut count = 0.0;
        for n in &neighbors {
            if let TileType::Resource(e) = self.get_tile(*n) { total += e; }
            count += 1.0;
        }
        if let TileType::Resource(e) = self.get_tile(pos) { total += e; count += 1.0; }
        total / count
    }

    pub fn tiles_ref(&self) -> &[TileType] { &self.tiles }
    pub fn dimensions(&self) -> (i32, i32) { (self.width, self.height) }
}

impl SubstrateInterface for Grid {
    fn sense(&self, position: Position) -> LocalState {
        LocalState {
            center: self.get_tile(position),
            north: self.get_tile(position.neighbor(Direction::North)),
            south: self.get_tile(position.neighbor(Direction::South)),
            east: self.get_tile(position.neighbor(Direction::East)),
            west: self.get_tile(position.neighbor(Direction::West)),
            resource_density: self.resource_density_at(position),
        }
    }

    fn is_walkable(&self, position: Position) -> bool { self.get_tile(position).is_walkable() }

    fn try_move(&mut self, _from: Position, to: Position) -> ActionResult {
        if self.is_walkable(to) { ActionResult::Moved(to) } else { ActionResult::Blocked }
    }

    fn try_consume(&mut self, position: Position) -> ActionResult {
        match self.get_tile(position) {
            TileType::Resource(energy) => { self.set_tile(position, TileType::Empty); ActionResult::Consumed(energy) }
            _ => ActionResult::NoEffect,
        }
    }

    fn tick(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.tiles.len() {
            if self.tiles[i] == TileType::Empty && rng.gen::<f32>() < self.resource_regen_rate {
                self.tiles[i] = TileType::Resource(self.resource_max);
            }
        }
    }

    fn width(&self) -> i32 { self.width }
    fn height(&self) -> i32 { self.height }
}
