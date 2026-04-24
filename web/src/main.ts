import { Renderer } from "./renderer";
import { WorldSnapshot } from "./types";

const canvas = document.getElementById("world") as HTMLCanvasElement;
const renderer = new Renderer(canvas);

const tickEl = document.getElementById("tick")!;
const orgCountEl = document.getElementById("org-count")!;
const resCountEl = document.getElementById("res-count")!;
const maxGenEl = document.getElementById("max-gen")!;

let latestSnapshot: WorldSnapshot | null = null;
let hasResized = false;

function connectWebSocket() {
  const ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onmessage = (event) => {
    try {
      latestSnapshot = JSON.parse(event.data) as WorldSnapshot;
    } catch {}
  };

  ws.onclose = () => {
    console.log("WebSocket closed, reconnecting in 2s...");
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => { ws.close(); };
}

function renderLoop() {
  if (latestSnapshot) {
    if (!hasResized) {
      renderer.resize(latestSnapshot.grid_width, latestSnapshot.grid_height);
      hasResized = true;
    }
    renderer.render(latestSnapshot);

    tickEl.textContent = latestSnapshot.tick.toString();
    orgCountEl.textContent = latestSnapshot.organism_count.toString();
    resCountEl.textContent = latestSnapshot.resource_count.toString();

    const maxGen = latestSnapshot.organisms.reduce((max, o) => Math.max(max, o.generation), 0);
    maxGenEl.textContent = maxGen.toString();

    const coverageEl = document.getElementById("coverage");
    if (coverageEl) {
      coverageEl.textContent = `${latestSnapshot.archive_coverage}/${latestSnapshot.archive_capacity}`;
    }

    const sessionsEl = document.getElementById("sessions");
    if (sessionsEl) sessionsEl.textContent = latestSnapshot.total_sessions.toString();
    const councilEl = document.getElementById("council");
    if (councilEl) councilEl.textContent = latestSnapshot.council_size.toString();
  }
  requestAnimationFrame(renderLoop);
}

connectWebSocket();
renderLoop();

// User input handling
const inputEl = document.getElementById("user-input") as HTMLInputElement;
const sendBtn = document.getElementById("send-btn")!;
const responseEl = document.getElementById("response")!;

sendBtn.addEventListener("click", async () => {
  const text = inputEl.value.trim();
  if (!text) return;

  // Show thinking state
  sendBtn.setAttribute("disabled", "true");
  responseEl.textContent = "Organism is thinking...";
  inputEl.value = "";

  try {
    const res = await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    responseEl.textContent = data.message;
  } catch {
    responseEl.textContent = "Failed to send message";
  } finally {
    sendBtn.removeAttribute("disabled");
  }
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});
