# OrganicAI

## Project Vision

A fundamentally new kind of AI that learns organically — without training data,
without model weights, without massive compute. A living digital organism that
grows its own brain, learns through experience, evolves across generations, and
becomes a daily companion that can research, solve problems, and help with coding.

Not a neural network. Not a trained model. A living digital organism.
One of a kind.

## Core Principles — NON-NEGOTIABLE

1. **No training data** — the brain learns through experience, not datasets
2. **No model weights** — neural structure grows from developmental programs
3. **Local computation only** — no backpropagation, no global loss functions
4. **No hardcoded logic** — no parsers, no dictionaries, no string matching,
   no if/else routing on content, no static text translations
5. **Intrinsic curiosity** — the brain thinks for itself when idle
6. **Substrate independence** — designed to migrate to quantum/neuromorphic hardware
7. **Everything the brain knows comes from experience** — Claude teaches, brain learns

If you're about to write `contains("solve")` or `match op { '+' => ...` or
`HashMap<String, String>` — STOP. That violates the principles. The brain
must discover structure from its own neural dynamics, not from code you write.

## Architecture

### Brain Components (crates/neuron/src/)

- **brain.rs** — OrganicBrain: 80M spiking neurons (LIF) with STDP, rayon parallelized
- **hdc.rs** — HDCMemory: 10,000-bit hypervectors, one-shot learning, 10,000+ fact capacity,
  compositional (BIND/BUNDLE/ROTATE), analogical reasoning
- **ring.rs** — NumberRing: ring attractor for math through bump dynamics
- **lif.rs** — Leaky Integrate-and-Fire neuron model
- **stdp.rs** — Spike-Timing-Dependent Plasticity learning rule
- **curiosity.rs** — Information gain, prediction error, homeostatic drive
- **lsm.rs** — Liquid State Machine readout from spiking reservoir
- **predictive.rs** — Predictive coding layers (free-energy prediction error)
- **attention.rs** — Gain modulation (organic attention, not transformer attention)
- **thinking.rs** — Chain recall (reasoning), context tracking, creative perturbation
- **inner_life.rs** — Brain daydreams when idle, discovers transitive knowledge
- **working_memory.rs** — 8 neural registers for step-by-step reasoning
- **memory.rs** — Legacy Hebbian attractor memory (replaced by HDC)

### Other Crates

- **core** — Cell, Genome, Organism, Position data types
- **substrate** — Grid world, SAL trait (Substrate Abstraction Layer)
- **growth** — Growth program interpreter (organisms grow from seed cells)
- **engine** — Simulation loop, energy economics, reproduction, persistence
- **evolution** — QD archive (MAP-Elites), behavioral descriptors, novelty bonus
- **tools** — Tool tiles (memory, pattern, logic, language, search, LLM, filesystem)
- **network** — Distributed brain across machines (TCP spike sharing, weight sync)
- **server** — WebSocket server, HTTP API, browser visualizer

### Query Processing Flow

```
Question arrives at brain.process()
    ↓
HDC fast recall (10,000-bit hypervector similarity, <1ms)
    ↓ found?
YES → return answer from brain's own memory
    ↓ no
Spiking network deep processing (80M neurons, attention-modulated)
    ↓ got something?
YES → return spiking network output
    ↓ no
Return empty → server calls Claude CLI → Claude answers → brain.train() stores in HDC
    ↓
Next time same question → brain answers from HDC, no Claude needed
```

### Training Flow

```
Claude answers a question the brain didn't know
    ↓
brain.train(question, answer)
    ↓
Compute surprise (HDC vector similarity between predicted and actual)
    ↓ surprising?
YES → store in HDC memory (one-shot, <1ms)
NO → skip (brain already knew this)
    ↓
Record in context, feed inner life
```

## Deployment

### Local (M4 MacBook, 16GB)
```bash
cargo run --release    # 40M neurons
# http://localhost:3000
```

### Remote (Linux, 32GB)
```bash
ssh fazle@192.168.2.2
cd ~/organic-ai
# Set 80M neurons (sed commands on brain.rs constants)
cargo build --release
nohup cargo run --release > /tmp/organic-ai.log 2>&1 &
# http://192.168.2.2:3000
```

### Training (zero Claude tokens after first run)
```bash
# Generate teaching files (one-time Claude cost):
bash train_smart.sh
# Files saved to training_data/*.txt (1,046 items)
# All subsequent rounds reuse files — free forever
```

## What Works Today

- 80M spiking neurons with rayon parallelism on 32GB Linux box
- HDC memory: 10,000+ fact capacity, one-shot learning, no cross-contamination
- Ring attractor: 100% accurate math (add, subtract, multiply, divide, power)
- Brain recalls learned facts from HDC vectors, not Claude
- Saves to disk, persists across restarts
- Browser visualizer at http://host:3000
- User text input via POST /api/message
- Inner life: brain thinks when idle, discovers transitive knowledge
- Organism simulation: growth, evolution, QD archive running continuously

## What's Next

- CLI tool for direct brain interaction (organic-ai "clean my computer")
- MCP server to learn from every Claude conversation passively
- Brain researches on its own when asked a question it can't answer
- Predictive coding for genuine hierarchical learning
- Cortical columns (1M structured neurons beats 80M random ones)
- Sequence learning for understanding sentences and code
- Earned autonomy for file operations (human approves, brain learns preferences)

## Context Engine (CCE)

This project uses Claude Context Engine for intelligent code retrieval.

**IMPORTANT: You MUST use `context_search` instead of reading files directly**
when exploring the codebase, answering questions about code, or understanding
how things work.

**When to use `Read` instead:**
- You need to edit a specific file (read before editing)
- You need the exact, complete content of a known file path

## Output Style

Be concise. Lead with the answer or action, not reasoning. Skip filler words,
preamble, and phrases like "I'll help you with that" or "Certainly!". Prefer
fragments over full sentences in explanations. No trailing summaries of what
you just did. One sentence if it fits.

Code blocks, file paths, commands, and error messages are always written in full.
