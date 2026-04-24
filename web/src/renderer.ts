import { WorldSnapshot } from "./types";

const CELL_TYPE_COLORS: Record<string, string> = {
  Sensor: "#4CAF50",
  Motor: "#F44336",
  Inter: "#2196F3",
  Reproductive: "#FF9800",
  Undifferentiated: "#9E9E9E",
};

const TOOL_COLORS: Record<string, string> = {
  Memory: "#E040FB",    // purple
  Pattern: "#00BCD4",   // cyan
  Logic: "#FF5722",     // deep orange
  Language: "#CDDC39",  // lime
  Search: "#03A9F4",     // light blue
  LLM: "#9C27B0",        // deep purple
  FileSystem: "#795548",  // brown
};

const BG_COLOR = "#1a1a2e";

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private cellSize: number;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.cellSize = 6;
  }

  resize(gridWidth: number, gridHeight: number) {
    this.cellSize = Math.max(
      2,
      Math.min(
        Math.floor(window.innerWidth * 0.8 / gridWidth),
        Math.floor(window.innerHeight * 0.8 / gridHeight)
      )
    );
    this.canvas.width = gridWidth * this.cellSize;
    this.canvas.height = gridHeight * this.cellSize;
  }

  render(snapshot: WorldSnapshot) {
    const { ctx, cellSize } = this;
    const w = snapshot.grid_width * cellSize;
    const h = snapshot.grid_height * cellSize;

    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, w, h);

    // Draw tool tiles
    if (snapshot.tool_positions) {
      for (const tool of snapshot.tool_positions) {
        ctx.fillStyle = TOOL_COLORS[tool.tool_type] || "#888";
        ctx.globalAlpha = 0.4;
        ctx.fillRect(tool.x * cellSize, tool.y * cellSize, cellSize, cellSize);
        ctx.globalAlpha = 1.0;
      }
    }

    for (const org of snapshot.organisms) {
      for (const cell of org.cells) {
        let color = CELL_TYPE_COLORS[cell.cell_type] || "#FFFFFF";
        if (cell.spike_active) {
          color = "#FFFFFF";
        }
        ctx.fillStyle = color;
        ctx.fillRect(cell.x * cellSize, cell.y * cellSize, cellSize - 1, cellSize - 1);

        if (cell.information_gain > 0.3) {
          ctx.fillStyle = `rgba(255, 255, 0, ${cell.information_gain * 0.3})`;
          ctx.fillRect(cell.x * cellSize, cell.y * cellSize, cellSize - 1, cellSize - 1);
        }
      }

      if (org.cells.length > 1) {
        const xs = org.cells.map(c => c.x);
        const ys = org.cells.map(c => c.y);
        const minX = Math.min(...xs);
        const minY = Math.min(...ys);
        const maxX = Math.max(...xs);
        const maxY = Math.max(...ys);

        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.strokeRect(
          minX * cellSize - 1,
          minY * cellSize - 1,
          (maxX - minX + 1) * cellSize + 2,
          (maxY - minY + 1) * cellSize + 2
        );
      }
    }
  }
}
