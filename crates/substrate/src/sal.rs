use organic_core::direction::Position;
use crate::tile::TileType;

#[derive(Debug, Clone)]
pub struct LocalState {
    pub center: TileType,
    pub north: TileType,
    pub south: TileType,
    pub east: TileType,
    pub west: TileType,
    pub resource_density: f32,
}

#[derive(Debug, Clone)]
pub enum ActionResult {
    Moved(Position),
    Consumed(f32),
    Blocked,
    NoEffect,
}

pub trait SubstrateInterface {
    fn sense(&self, position: Position) -> LocalState;
    fn is_walkable(&self, position: Position) -> bool;
    fn try_move(&mut self, from: Position, to: Position) -> ActionResult;
    fn try_consume(&mut self, position: Position) -> ActionResult;
    fn tick(&mut self);
    fn width(&self) -> i32;
    fn height(&self) -> i32;
}
