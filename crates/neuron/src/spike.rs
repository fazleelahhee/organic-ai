use organic_core::cell::{Cell, CellId};
use organic_core::direction::Direction;

pub fn collect_synaptic_input(cell: &Cell, pre_spikes: &[(CellId, bool)]) -> f32 {
    let mut total = 0.0;
    for synapse in &cell.connections {
        for &(id, spiking) in pre_spikes {
            if id == synapse.target && spiking {
                total += synapse.weight;
            }
        }
    }
    total
}

pub fn decode_motor_output(recent_spikes: &[bool]) -> Option<Direction> {
    if recent_spikes.is_empty() { return None; }
    let pattern_value: u32 = recent_spikes.iter().enumerate()
        .filter(|(_, &s)| s)
        .map(|(i, _)| 1u32 << (i % 4))
        .fold(0u32, |acc, v| acc | v);
    if pattern_value == 0 { return None; }
    let dirs = Direction::all();
    Some(dirs[(pattern_value as usize) % 4])
}

#[cfg(test)]
mod tests {
    use super::*;
    use organic_core::cell::{Cell, CellType, Synapse};
    use organic_core::direction::Position;

    fn make_cell(id: u64, ct: CellType) -> Cell {
        let mut c = Cell::new_seed(id, Position::new(0, 0));
        c.cell_type = ct;
        c
    }

    #[test]
    fn test_collect_inputs_from_spiking_presynaptic() {
        let mut post = make_cell(1, CellType::Inter);
        post.connections.push(Synapse { target: 0, weight: 0.5, last_pre_tick: None });
        let input = collect_synaptic_input(&post, &[(0u64, true)]);
        assert_eq!(input, 0.5);
    }

    #[test]
    fn test_no_input_when_pre_not_spiking() {
        let mut post = make_cell(1, CellType::Inter);
        post.connections.push(Synapse { target: 0, weight: 0.5, last_pre_tick: None });
        let input = collect_synaptic_input(&post, &[(0u64, false)]);
        assert_eq!(input, 0.0);
    }

    #[test]
    fn test_multiple_inputs_sum() {
        let mut post = make_cell(2, CellType::Inter);
        post.connections.push(Synapse { target: 0, weight: 0.3, last_pre_tick: None });
        post.connections.push(Synapse { target: 1, weight: 0.4, last_pre_tick: None });
        let input = collect_synaptic_input(&post, &[(0u64, true), (1u64, true)]);
        assert!((input - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_decode_spike_pattern_picks_direction() {
        let dir = decode_motor_output(&[true, false, true, false]);
        assert!(matches!(dir, Some(_)));
    }
}
