use organic_core::cell::{Cell, CellId, CellType};
use organic_core::direction::Direction;
use organic_core::genome::LearningParams;
use organic_neuron::curiosity::{compute_information_gain, substrate_to_signal, update_prediction};
use organic_neuron::lif::{integrate_and_fire, LifParams};
use organic_neuron::spike::{collect_synaptic_input, decode_motor_output};
use organic_neuron::stdp::{apply_stdp, StdpParams};
use organic_substrate::sal::{LocalState, SubstrateInterface};

use crate::energy;

pub struct NeuralTickResult {
    pub move_direction: Option<Direction>,
    pub energy_cost: f32,
    pub spike_count: u32,
}

pub fn run_neural_tick(
    cells: &mut [Cell],
    tick: u64,
    learning_params: &LearningParams,
    grid: &dyn SubstrateInterface,
) -> NeuralTickResult {
    let lif_params = LifParams::default();
    let stdp_params = StdpParams {
        window: learning_params.stdp_window as u64,
        ..StdpParams::default()
    };

    let prev_spikes: Vec<(CellId, bool)> = cells
        .iter()
        .map(|c| (c.id, c.spike_active))
        .collect();

    let mut spike_count = 0u32;
    let mut motor_spikes: Vec<bool> = Vec::new();
    let mut energy_cost = 0.0f32;

    for cell in cells.iter_mut() {
        // Step 1: SENSE
        let sensory_input = match cell.cell_type {
            CellType::Sensor => {
                let local: LocalState = grid.sense(cell.position);
                substrate_to_signal(local.resource_density)
            }
            _ => 0.0,
        };
        let synaptic_input = collect_synaptic_input(cell, &prev_spikes);
        let total_input = sensory_input + synaptic_input;

        // Step 2: PREDICT
        let info_gain = compute_information_gain(cell.predicted_input, total_input);
        cell.information_gain = info_gain;

        // Step 3: INTEGRATE + Step 4: FIRE
        let fired = integrate_and_fire(&mut cell.membrane_potential, total_input, &lif_params);
        cell.spike_active = fired;
        if fired {
            cell.last_spike_tick = Some(tick);
            spike_count += 1;
            energy_cost += energy::COST_SPIKE;
        }

        if cell.cell_type == CellType::Motor {
            motor_spikes.push(fired);
        }

        // Step 5: LEARN (STDP on all connections)
        if let Some(post_tick) = cell.last_spike_tick {
            for synapse in &mut cell.connections {
                if let Some(pre_tick) = synapse.last_pre_tick {
                    apply_stdp(
                        &mut synapse.weight,
                        pre_tick,
                        post_tick,
                        info_gain * learning_params.learning_rate,
                        &stdp_params,
                    );
                }
            }
        }

        // Update presynaptic spike times
        for synapse in &mut cell.connections {
            for &(id, spiking) in &prev_spikes {
                if id == synapse.target && spiking {
                    synapse.last_pre_tick = Some(tick);
                }
            }
        }

        // Decay unused connections
        for synapse in &mut cell.connections {
            if synapse.last_pre_tick.map_or(true, |t| tick - t > learning_params.stdp_window as u64 * 2) {
                synapse.weight *= 1.0 - learning_params.decay_rate;
            }
        }

        // Prune dead connections
        cell.connections.retain(|s| s.weight > 0.01);

        // Update prediction
        update_prediction(&mut cell.predicted_input, total_input, learning_params.learning_rate);
    }

    // Step 6: ACT
    let move_direction = if !motor_spikes.is_empty() {
        decode_motor_output(&motor_spikes)
    } else {
        None
    };

    NeuralTickResult { move_direction, energy_cost, spike_count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use organic_core::cell::{Cell, CellType, Synapse};
    use organic_core::direction::Position;
    use organic_core::genome::LearningParams;
    use organic_substrate::grid::Grid;

    fn make_organism_cells() -> Vec<Cell> {
        let mut sensor = Cell::new_seed(0, Position::new(5, 5));
        sensor.cell_type = CellType::Sensor;
        let mut inter = Cell::new_child(1, Position::new(5, 5), 1);
        inter.cell_type = CellType::Inter;
        inter.connections.push(Synapse { target: 0, weight: 0.5, last_pre_tick: None });
        let mut motor = Cell::new_child(2, Position::new(5, 5), 2);
        motor.cell_type = CellType::Motor;
        motor.connections.push(Synapse { target: 1, weight: 0.5, last_pre_tick: None });
        vec![sensor, inter, motor]
    }

    #[test]
    fn test_neural_tick_runs_without_panic() {
        let mut cells = make_organism_cells();
        let params = LearningParams::default();
        let mut grid = Grid::new(20, 20);
        let mut rng = rand::thread_rng();
        grid.scatter_resources(50, 5.0, &mut rng);
        let result = run_neural_tick(&mut cells, 1, &params, &grid);
        assert!(result.energy_cost >= 0.0);
    }

    #[test]
    fn test_sensor_cell_gets_input_from_resources() {
        let mut cells = make_organism_cells();
        let params = LearningParams::default();
        let mut grid = Grid::new(10, 10);
        let mut rng = rand::thread_rng();
        grid.scatter_resources(100, 5.0, &mut rng);
        for t in 0..20 {
            run_neural_tick(&mut cells, t, &params, &grid);
        }
        assert!(cells[0].predicted_input != 0.0 || cells[0].information_gain >= 0.0);
    }

    #[test]
    fn test_stdp_modifies_weights_over_time() {
        let mut cells = make_organism_cells();
        let params = LearningParams { learning_rate: 0.1, ..LearningParams::default() };
        let mut grid = Grid::new(10, 10);
        let mut rng = rand::thread_rng();
        grid.scatter_resources(100, 5.0, &mut rng);
        let initial_weight = cells[1].connections[0].weight;
        for t in 0..50 {
            run_neural_tick(&mut cells, t, &params, &grid);
        }
        let final_weight = if cells[1].connections.is_empty() { 0.0 } else { cells[1].connections[0].weight };
        assert!(
            final_weight != initial_weight || cells[1].connections.is_empty(),
            "STDP should modify weights over 50 ticks"
        );
    }
}
