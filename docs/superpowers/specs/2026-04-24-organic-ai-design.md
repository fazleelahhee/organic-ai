# OrganicAI — Design Specification

**Date:** 2026-04-24
**Status:** Draft
**Author:** Fazlee + Claude (companion)

---

## Vision

A fundamentally new kind of AI that learns organically — without training data, without model weights, without massive compute. An AI that grows its own brain from a seed cell, learns through curiosity, evolves across generations, and eventually becomes a daily companion that can research, solve problems, and handle tasks.

Not a neural network. Not a trained model. A living digital organism.

---

## Core Principles

1. **No training** — organisms learn through experience, not datasets
2. **No model** — neural structure grows from a developmental program, not a fixed architecture
3. **Local only** — every computation is local. No global loss function, no backpropagation
4. **Intrinsic curiosity** — organisms seek novelty because their neurons are wired to maximize information gain
5. **Substrate independence** — the system runs on a classical grid today but is designed to migrate to quantum, neuromorphic, or photonic substrates without re-engineering the organisms
6. **Start small, scale infinitely** — petri dish to ecosystem to companion, same architecture throughout

---

## Architecture: "Infomorphic Embryogenesis"

Combines two cutting-edge paradigms:

- **Neural Developmental Programs** (Risi lab, 2023-24) — organisms grow from a seed cell via a genetic growth program
- **Infomorphic Neurons** (Uni Gottingen / Max Planck, PNAS 2025) — each neuron is an autonomous agent that learns independently using information theory

### The Three Nested Loops

```
EVOLUTION (across generations)
  └── DEVELOPMENT (organism lifetime, early phase)
        └── LEARNING (organism lifetime, continuous)
```

- **Evolution** discovers growth programs and learning parameters that produce capable organisms
- **Development** executes the growth program — a seed cell divides, differentiates, and wires itself into a neural body
- **Learning** happens continuously — each cell adapts its connections via STDP, driven by intrinsic curiosity (information gain)

---

## Terminology

| Term | Meaning |
|---|---|
| **Cell** | Atomic unit. An autonomous infomorphic neuron with position, state, connections, and its own learning objective. |
| **Genome** | Compact data structure encoding: growth program + learning parameters + intrinsic drives + substrate hints. The DNA. |
| **Organism** | Colony of cells grown from a single seed cell. No central controller — intelligence is distributed. |
| **Substrate** | The world where organisms live. 2D grid (initially), abstracted behind the SAL. |
| **Spike** | Discrete communication signal between cells. Timing matters (STDP). |
| **Information Gain** | Each cell's intrinsic reward — how surprising/novel its input was. Drives learning and curiosity. |
| **Growth Program** | Local rules in the genome that tell a seed cell how to divide, differentiate, and form connections. |
| **SAL** | Substrate Abstraction Layer. Interface between organisms and their computational backend. Enables substrate independence. |
| **Lineage** | Evolutionary history of a genome across generations. |

---

## The Four Layers

```
+---------------------------------------------------+
|           LAYER 4: ECOSYSTEM                       |
|  Multiple organisms, co-evolution, competition,    |
|  cooperation, predator/prey, resource dynamics     |
+---------------------------------------------------+
|           LAYER 3: ORGANISM                        |
|  Colony of cells grown from genome. Distributed    |
|  intelligence. Senses, acts, feeds, reproduces.    |
+---------------------------------------------------+
|           LAYER 2: CELL (Infomorphic Neuron)       |
|  Autonomous agent. Spikes. STDP. Info-theoretic    |
|  learning. Intrinsic curiosity.                    |
+---------------------------------------------------+
|           LAYER 1: SUBSTRATE                       |
|  Grid world -> continuous -> 3D -> quantum.        |
|  Resources, physics, SAL.                          |
+---------------------------------------------------+
```

### Layer 1: Substrate

The world where everything happens.

- **Grid** — 2D array of tiles. Types: empty, resource (energy), hazard, wall
- **Resources** regenerate over time — creates foraging pressure
- **Physics** are local: cells occupy tiles, sense neighbors, move one step at a time
- **SAL interface:**
  - `sense(position) -> local_state`
  - `act(action) -> result`
  - `tick()` — advance world state
- **Substrate independence:** today runs on a Rust grid engine. Same interface maps to quantum lattice, continuous physics, neuromorphic chip, or photonic processor in the future. Organisms never interact with the backend directly.

### Layer 2: Cell (Infomorphic Neuron)

Every cell is an autonomous agent.

**State:**
- Position on substrate
- Membrane potential (Leaky Integrate-and-Fire model)
- Connections to other cells (synapses with weights)
- Cell type: sensor, inter, motor, reproductive
- Local learning parameters (inherited from genome, fine-tuned by experience)

**Behavior each tick (6-step cycle):**

1. **Sense** — receive spikes from connected cells + substrate input (if sensor)
2. **Predict** — what did I expect? Compute prediction error.
3. **Integrate** — update membrane potential from inputs
4. **Fire** — if threshold crossed, emit spike to all outgoing synapses
5. **Learn** — STDP adjusts weights, modulated by information gain. Prune dead connections, grow new ones toward surprising neighbors.
6. **Act** — motor cells translate spike patterns into substrate actions

**Curiosity mechanism:**
Each cell maintains a prediction of its next input. Information gain = prediction error. High gain strengthens contributing connections. Cells naturally wire toward novel, surprising signals. Curiosity isn't a reward — it's how the wiring works.

### Layer 3: Organism

A colony of cells, not a monolith.

**Birth:** Single seed cell placed on substrate with a genome.

**Development phase:**
1. Seed cell divides -> two cells with a connection
2. Each cell checks growth program: divide again? Differentiate? Form long-range connections?
3. Growth rules are conditioned on: cell age, neighbor count, local resource density, neighbor signals, depth from seed
4. Development completes when growth program halts or max steps reached

**Living phase:**
- Sensor cells read substrate -> spikes propagate -> motor cells act
- All cells learn continuously via STDP + information gain
- Organism moves, feeds, explores
- Energy budget: every tick costs energy. Starvation = death.

**Reproduction:**
- Energy exceeds threshold -> copy genome with mutations -> new seed cell nearby
- Offspring inherits DNA, not the parent's learned connections

**Death:**
- Energy reaches zero -> cells deactivate -> space freed

### Layer 4: Ecosystem

- Multiple organisms coexist on the substrate
- Quality-diversity archive tracks interesting genomes across behavioral dimensions
- Novelty pressure prevents collapse to one dominant strategy
- Co-evolution: organisms become part of each other's environment

---

## The Genome

```
+-----------------------------------------------------+
|                      GENOME                          |
|                                                      |
|  SECTION A: Growth Program                           |
|  - Division rules (when/how cells divide)            |
|  - Differentiation rules (cell type assignment)      |
|  - Connection rules (synapse formation)              |
|  - Termination conditions                            |
|                                                      |
|  SECTION B: Learning Parameters                      |
|  - STDP time window (5-50 ticks)                     |
|  - Learning rate per cell type (0.001-0.1)           |
|  - Information gain sensitivity (0.0-1.0)            |
|  - Prediction depth (1-5 ticks)                      |
|  - Decay rate for unused connections (0.0-0.05)      |
|  - Max connections per cell (3-20)                   |
|                                                      |
|  SECTION C: Intrinsic Drives                         |
|  - Curiosity weight (novelty-seeking strength)       |
|  - Hunger sensitivity (energy-seeking calibration)   |
|  - Social drive (attraction/repulsion to others)     |
|  - Homeostatic targets (preferred internal states)   |
|                                                      |
|  SECTION D: Substrate Hints (dormant)                |
|  - Computational primitive preferences               |
|  - Inert until new substrates available              |
|                                                      |
+-----------------------------------------------------+
```

### Growth Program Rules

Conditional local rules of the form: `IF <condition> THEN <action>`

**Conditions:** cell age, neighbor count, resource density, parent signal, depth from seed

**Actions:** divide(direction), differentiate(type), connect(target_rule), emit_signal(value), halt

**Execution order:** Rules execute sequentially top-to-bottom each tick. First matching rule fires. One action per cell per tick.

**Example:**
```
IF age > 3 AND neighbor_count < 4 THEN divide(random_direction)
IF depth_from_seed > 5 AND resource_density < 0.3 THEN differentiate(sensor)
IF depth_from_seed > 8 THEN differentiate(motor)
IF age > 10 THEN connect(nearest, type=inter, max_distance=3)
IF neighbor_count > 6 THEN halt
```

### Intrinsic Drives

Curiosity is a homeostatic drive, not a reward signal. The organism has a target level of information gain. Too predictable = discomfort, seek novelty. Too chaotic = discomfort, seek stability. This mirrors biological brains.

### Mutation

| Type | Effect |
|---|---|
| Point mutation | Change a single numeric parameter +/- small delta |
| Rule insertion | Add a new growth rule |
| Rule deletion | Remove a growth rule |
| Rule swap | Reorder two growth rules |
| Crossover | Swap sections between two genomes (if sexual reproduction evolves) |

Mutation rates are themselves evolvable.

---

## Simulation Loop — One World Tick

```
1. SUBSTRATE UPDATE
   - Regenerate resources
   - Apply physics (hazard spread, decay)

2. FOR EACH ORGANISM (parallel):
   IF development phase:
     - Execute next growth program step
     - Divide / differentiate / connect
     - Check termination
   IF living phase:
     FOR EACH CELL (parallel):
       a. Sense (substrate + incoming spikes)
       b. Predict (what did I expect?)
       c. Integrate (update membrane potential)
       d. Fire? (threshold -> spike)
       e. Learn (STDP + info gain)
       f. Act (motor cells -> substrate)
     - Deduct energy cost
     - Check reproduction threshold
     - Check death (energy <= 0)

3. REPRODUCTION
   - Copy genome with mutations
   - Place seed cell near parent
   - Register in QD archive

4. DEATH & CLEANUP
   - Remove dead organisms
   - Free substrate tiles

5. ECOSYSTEM METRICS (every N ticks)
   - Update diversity map
   - Compute population stats
   - Emit visualization data
```

### Energy Economics

| Action | Energy Cost |
|---|---|
| Existing (per cell, per tick) | -0.01 |
| Sensing | -0.005 |
| Firing a spike | -0.02 |
| Moving | -0.1 |
| Growing a new cell | -0.5 |
| Reproducing | -50% of current energy |
| Consuming resource tile | +5.0 |

### Parallelism

- Organisms are independent — tick in parallel (rayon)
- Cells within an organism use 1-tick spike delay — process in parallel using last tick's spikes
- Substrate updates are grid-local — parallelizable
- Thousands of organisms with hundreds of cells each can tick in real-time on 8-16 cores

---

## External World Interface — Tool Use as Evolved Ability

### Three Environments (Progressive Unlocking)

**Environment 1: Grid World (M1-M3)**
- Pure simulation. Organisms learn survival, foraging, social behavior.

**Environment 2: Tools World (M4)**
- Abstract tool tiles appear on the substrate
- Memory tile — store/retrieve signal patterns
- Pattern tile — compare patterns, get similarity
- Logic tile — boolean/arithmetic on signal patterns
- Language tile — tokenize/detokenize signals to/from text
- Challenge environments — puzzles requiring tool use

**Environment 3: Real World Interface (M5-M6)**
- Tool tiles connect to actual external services through the SAL:

| Tool Tile | Backend | Organism Experience |
|---|---|---|
| Search tile | Web search API | Emit signal -> receive information pattern |
| LLM tile | Claude / Codex API | Emit query -> receive reasoning pattern |
| Task tile | Calendar, email, files | Emit action -> receive confirmation |
| Code tile | Sandboxed execution | Emit program -> receive output |

### How Tool Use Evolves

No organism is taught to use tools. The three loops discover tool use:

1. **Evolution** — organisms near tool tiles sometimes benefit. Genomes that grow tool-oriented sensors get selected.
2. **Development** — growth programs evolve "tool appendages" — specialized cell clusters for tool interaction.
3. **Learning** — Hebbian/infomorphic learning strengthens pathways that led to successful tool use. Information gain is high when tools return useful data.

### Safety

| Principle | Implementation |
|---|---|
| Sandboxed execution | All actions go through SAL — no bypass |
| Action budget | External actions cost energy — can't spam APIs |
| Capability gating | Real-world tiles unlock after demonstrated competence |
| Human approval | High-stakes actions require confirmation initially |
| Reversibility preference | Read-only tools cost less than write tools |
| Earned autonomy | Approval requirements relax as trust builds |

---

## Tech Stack

### Language Responsibilities

| Language | Role | Why |
|---|---|---|
| **Rust** | Core simulation engine | Performance, safety, parallelism (rayon) |
| **Python** | Research, evolution, external APIs | Ecosystem (pyribs, numpy, Anthropic SDK) |
| **JavaScript/TS** | Live visualization, user interface | Browser-based, interactive, shareable |

### Project Structure

```
organic_ai/
├── crates/
│   ├── core/              # Cell, Organism, Genome data structures
│   ├── substrate/         # Grid world, SAL interface, physics
│   ├── neuron/            # LIF model, STDP, infomorphic learning
│   ├── growth/            # Growth program interpreter
│   ├── evolution/         # Mutation engine, selection
│   ├── engine/            # Main simulation loop, tick orchestration
│   ├── tools/             # Tool tiles, external world interface
│   ├── server/            # WebSocket server, API endpoints
│   └── pyo3-bindings/     # Python FFI layer
├── python/
│   ├── organic_ai/        # Python package wrapping Rust
│   ├── experiments/       # Experiment scripts
│   └── notebooks/         # Jupyter notebooks
├── web/
│   ├── src/               # TypeScript visualization
│   ├── shaders/           # WebGPU shaders for grid rendering
│   └── index.html
├── docs/
│   └── superpowers/
│       └── specs/
└── Cargo.toml
```

### Key Dependencies

| Component | Library | Purpose |
|---|---|---|
| Parallelism | rayon | Data-parallel cell ticking |
| Async | tokio | WebSocket server, API calls |
| Serialization | serde + bincode | Genome/state serialization |
| WebSocket | axum | Browser communication |
| Python FFI | pyo3 + maturin | Rust -> Python bridge |
| QD Evolution | pyribs | MAP-Elites quality-diversity |
| Visualization | Canvas + WebGPU | Grid rendering |
| Charts | D3.js | Dashboards, lineage trees |
| LLM | anthropic SDK | Tool tile -> Claude API |

### Data Flow

```
Rust Engine (1000+ ticks/sec)
  |
  +-> State snapshot (every N ticks) -> WebSocket -> Browser
  +-> Metrics (every M ticks) -> Python (logging, analysis)
  +-> Tool requests (async) -> Python -> External APIs -> back to organism
```

---

## Milestone Roadmap

### M1: "It's Alive" — Self-Replicating Organisms

Build: core, substrate, growth, engine crates + JS grid visualizer
Prove: Growth programs work. Organisms self-assemble, eat, reproduce.
Success: 3+ generations survive. Real-time visualization.
Scope: ~3,000 lines Rust + ~500 lines JS

### M2: "It Learns" — Infomorphic Hebbian Cells

Build: neuron crate (LIF, STDP, infomorphic learning, curiosity)
Prove: Organisms improve within their lifetime. Learning without training.
Success: Measurable foraging improvement per lifetime. Learning > no-learning.
Scope: ~2,500 lines Rust + ~300 lines JS

### M3: "It Evolves" — Ecosystem & Quality-Diversity

Build: evolution crate, Python bindings, pyribs integration, ecosystem dashboard
Prove: Open-ended evolution. Multiple species. Emergent behaviors.
Success: 5+ distinct species coexist 10,000+ ticks. 50+ QD niches. At least one surprise.
Scope: ~2,000 lines Rust + ~1,500 lines Python + ~800 lines JS

### M4: "It Thinks" — Abstract Tool Use

Build: tools crate (memory, pattern, logic, language tiles), challenge environments
Prove: Organisms evolve tool use without being taught.
Success: 1+ lineage consistently uses 2+ tools. Tool-users outperform non-users.
Scope: ~2,000 lines Rust + ~1,000 lines Python + ~500 lines JS

### M5: "It Reaches Out" — Real World Interface

Build: external SAL backends, safety layer, natural language I/O, async tool execution
Prove: Organism can search the web, query Claude, answer questions.
Success: 10+ successful web-search answers. Response quality improves over time.
Scope: ~1,500 lines Rust + ~2,000 lines Python + ~1,000 lines JS

### M6: "It Lives Alongside You" — The Companion

Build: persistent state, proactive behavior, earned autonomy, multi-organism council
Prove: A living AI companion that knows you and grows with you.
Success: 30+ sessions with accumulated learning. 5+ daily task categories. Feels organic.
Scope: ~2,000 lines Rust + ~2,000 lines Python + ~1,500 lines JS

---

## Substrate Independence & Self-Adaptation

The system is designed from day one for substrate migration:

- **SAL** abstracts all organism-substrate interaction. Organisms don't know what hardware they run on.
- **Section D of the genome** (substrate hints) is dormant now but will activate when new computational primitives become available.
- **Infomorphic learning** uses information-theoretic objectives that translate directly to quantum information theory.
- **STDP** is a local rule — works identically whether the underlying signal is a classical spike or a quantum state.
- **Growth programs** are abstract — they say "divide, connect, specialize," not "use float32."

When quantum computers become generally available, the SAL gets a new backend. Evolution discovers organisms that exploit quantum primitives (superposition, entanglement) because those organisms gain an information-processing advantage. No human re-engineering needed.

No existing system has been designed with this principle. This is a first.

---

## Research Foundations

This design synthesizes cutting-edge work from multiple fields:

| Paradigm | Key Work | How We Use It |
|---|---|---|
| Neural Cellular Automata | Google Growing NCA (2020), Lenia | Self-organizing substrate inspiration |
| Neural Developmental Programs | Risi lab (2023-24), Lifelong NDP | Growth program architecture |
| Infomorphic Neurons | Uni Gottingen / MPI, PNAS 2025 | Autonomous cell learning |
| Hebbian/STDP | Whittington & Bogacz (2017) | Local synaptic plasticity |
| Forward-Forward | Hinton (2022), Self-Contrastive FF (2025) | Backprop-free learning validation |
| Quality-Diversity | MAP-Elites, pyribs | Evolution strategy |
| Novelty Search | Lehman & Stanley | Open-ended evolution |
| Spiking Neural Networks | BindsNET, Brian2, NeuEvo | Neuron model reference |
| Artificial Life | Tierra, Avida, ASAL (Sakana AI, 2024) | Ecosystem dynamics |
| Open-Ended Evolution | POET, Dominated Novelty Search (2025) | Preventing evolutionary stagnation |

**The novel synthesis:** No existing system combines infomorphic neurons + neural developmental programs + STDP + quality-diversity evolution + substrate independence + progressive tool use in a single architecture. Each piece exists in research. The combination is new.

---

## What This Is NOT

- **Not a neural network** — no fixed architecture, no weight matrices, no training loop
- **Not a language model** — doesn't predict tokens, doesn't have a prompt template
- **Not reinforcement learning** — no external reward signal, no policy optimization
- **Not a simulation toy** — the roadmap leads to a practical daily companion
- **Not dependent on massive compute** — designed to run on a laptop, scale optionally

---

## What This IS

A living digital organism that:
- Grows its own brain from a single cell
- Learns through curiosity, not training
- Evolves across generations, not update cycles
- Uses tools because it discovered them, not because they were given
- Becomes your companion because it adapted to you, not because it was prompted to

One of a kind.
