export interface ToolTileSnapshot {
  x: number;
  y: number;
  tool_type: string;
}

export interface WorldSnapshot {
  tick: number;
  grid_width: number;
  grid_height: number;
  organisms: OrganismSnapshot[];
  resource_count: number;
  organism_count: number;
  archive_coverage: number;
  archive_capacity: number;
  max_generation: number;
  tool_positions: ToolTileSnapshot[];
}

export interface OrganismSnapshot {
  id: number;
  cells: CellSnapshot[];
  energy: number;
  phase: string;
  generation: number;
  age: number;
}

export interface CellSnapshot {
  x: number;
  y: number;
  cell_type: string;
  spike_active: boolean;
  information_gain: number;
}
