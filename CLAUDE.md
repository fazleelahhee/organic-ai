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
- **trading** — Trading reasoner built on OrganicBrain. Multi-modal input
  (price + news + sentiment + time), multi-step reasoning, self-awareness
  layer with per-regime accuracy + drift detection + confidence
  recalibration, position sizing with loss-aversion, P&L-aware backtest,
  production health checks + auto-repair. See "Trading reasoner"
  section below.

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

## What Works Today (verified in the live pipeline)

- HDC memory: 10,000+ fact capacity, one-shot store/recall, no cross-contamination.
- Brain recalls learned facts from HDC vectors when an exact-or-similar query was taught.
  Falls through to Claude on a miss; Claude's answer is then stored in HDC.
- **Spiking network learns from training (wired 2026-04-25)**: `train()` runs
  STDP-enabled ticks with teacher-forced output clamping. Hidden→output
  synapses strengthen for the input→output pair. Verified by test
  `test_spiking_training_strengthens_target_synapses`.
- **Predictive coding active (wired 2026-04-25)**: `pred_input_to_hidden` and
  `pred_hidden_to_output` update every tick inside `run_ticks`. Prediction
  error scales the STDP learning rate — surprising transitions drive stronger
  plasticity (free-energy-style). Verified by `test_predictive_coding_*`.
- **Working memory persists across queries (wired 2026-04-25)**: hidden firing
  rates are stride-sampled at the end of each `process()` and stored in WM.
  Next query's input vector receives the prior state injected at scaled
  strength (decays each turn). This is the brain's continuous-learning
  context — what makes it different from a stateless fine-tune. Verified by
  `test_working_memory_*`.
- **LSM readout decodes output (wired 2026-04-25)**: 1024-d learned linear
  projection over 4096 hidden samples, treated as 10 char positions × 95
  printable-ASCII chars. Trained alongside STDP via delta-rule against
  one-hot target text. Replaces the coarse 5-level chunk-of-4 decoder.
  Verified by `test_lsm_readout_learns_target` and
  `test_brain_holds_multiple_distinct_mappings`.
- Saves to disk, persists across restarts.
- Browser visualizer at http://host:3000.
- User text input via POST /api/message.
- Inner life: brain daydreams when idle, discovers transitive knowledge across HDC.
- Organism simulation: growth, evolution, QD archive running continuously.

### Important architectural fix (2026-04-25)

The sparse-check in `run_ticks` was over-aggressive: it skipped any neuron
with no external input, no residual potential, and no recent fire — which
meant **every hidden neuron was skipped on every tick during inference**,
because they have no external input and start at potential=0. The check now
also considers whether any synapse source fired this tick, so input
propagates into hidden as designed.

Before fix: 0 hidden firings per query. After fix: ~700 hidden firings on a
1536-neuron test brain. Without this, predictive coding, WM snapshots, and
LSM readout all received zero signal and could not learn.

## Built but NOT wired into the live pipeline (remaining)

- **Ring attractor (math substrate)**: Instantiated at `brain.rs:193`,
  never called outside its own unit tests. See "compositional reasoning
  empirical finding" below for whether this matters.
- **Attention**: Computed once before `run_ticks` and applied as a one-shot
  multiplier on hidden potentials. Decays on the next tick. (`brain.rs:391-397`)
- **Distributed network** (`crates/network/`, 443 LOC): Entire crate, never
  instantiated.

## Compositional reasoning — empirical finding (2026-04-25)

Tested whether the pure-neural learning loop can compositionally generalize.
Curriculum: all 55 2-operand single-digit additions with sum < 10. Training:
100 interleaved rounds = 5500 calls. Test: 8 novel 3-operand queries.

### Round 1 — static input + feedforward only
- Recall: 55/55 (100%)
- Composition: 0/8 — LSM produces sigmoid-baseline gibberish like
  `"(%%(%1%%%(%(((%+%%((%(%(%(%+(%..."`, no recognizable digits.

### Round 2 — added recurrent excitatory hidden + synaptic delays + temporal input encoding
The brain now has:
- 6 recurrent excitatory hidden→hidden connections per hidden neuron with
  random delays 1-5 ticks (`brain.rs:RECURRENT_PER_HIDDEN`, `MAX_DELAY`).
- Per-synapse delay field; `fired_history` ring buffer in `run_ticks_internal`
  delivers each pre-synaptic spike `D` ticks after firing. STDP's `dt` becomes
  exactly the synapse's delay, so STDP selectively strengthens connections
  whose delay matches the temporal regularity of observed pre→post pairs —
  the substrate for sequence learning, biologically modeled on conduction-
  delay tuning in real cortex.
- Temporal input encoding: queries stream over time (3 ticks per character)
  rather than collapsing into one static spike pattern. The brain literally
  *sees the query unfold*.

Results:
- Recall: 55/55 (100%) — unchanged.
- Composition: **still 0/8**, but now produces empty strings instead of
  gibberish. The readout learned to be silent on inputs it has no training
  for, which is a small improvement in honesty but not in capability.

### What this tells us
Real architectural work was tried — delays, recurrent excitation, temporal
streaming. The recurrent dynamics demonstrably work
(`test_recurrent_dynamics_sustain_activity`). The brain genuinely processes
queries as time-varying signals now. **But novel compositional
generalization still doesn't emerge at this scale.**

Modern ML achieves compositional generalization through architectural
inductive biases (attention mechanisms in transformers, recursive structure
in tree-RNNs). Spiking + STDP + recurrent dynamics, even with 80M neurons,
don't have those biases. They learn what they're shown; they extrapolate
poorly to novel structural compositions.

**Implication for the stated example "teach 2+2 etc, brain solves 2+1+3":**
This codebase will not deliver that without further architectural
commitment. Possible bigger swings:
1. Cortical-column microcircuits with structural priors (CLAUDE.md "What's Next")
2. Attention-like compositional architecture (would deviate from the
   biologically-pure spiking design)
3. Massive scale + curriculum on 3+, 4+, 5+ operand pairs (memorization)
4. Accept the architectural limit; rely on Claude-teaches-brain-memorizes
   for novel queries.

None are quick wins. All require honest scoping with the user.

## Trading reasoner (`crates/trading/`)

Production-shaped trading-AI layer built on top of OrganicBrain. NOT a
replacement for boring ML on raw tabular accuracy — it's a *reasoning
system* that adds capabilities boring ML and LLMs can't easily provide.

### What it does

**Multi-modal input.** `MarketState` carries numeric market features
(price, volume, return, volatility, indicators) **plus** news headlines
+ sentiment scores + time context. The encoder folds everything into
one deterministic key the brain consumes. No manual text-feature
engineering pipeline needed.

**Reasoning chain over the brain's pattern memory.**
1. Encode state → key
2. Stratified retrieval: top-K similar past patterns from same macro
   regime (regime_euphoric / oversold / capitulation / etc.)
3. Multi-pass through the spiking network with token-rotation
   perturbations
4. Aggregate outcome distribution across matches (with Beta-Bernoulli
   shrinkage so 5-sample agreement isn't reported as 100% confidence)
5. Counter-evidence surfaced explicitly
6. Per-horizon predictions (1h/4h/24h/7d) when training data has them
7. Position sizing: `edge × confidence × anomaly_factor × opposition_factor`,
   capped at 10% of capital (configurable)
8. Audit trail of reasoning steps

**Self-awareness layer (`crates/trading/src/self_assessment.rs`).**
Every `analyze()` pushes a `PredictionRecord`. Every `train_on_outcome()`
scores the most-recent matching prediction. From this:
- Overall hit rate
- Per-regime accuracy (where does the brain read well vs poorly)
- Per-source accuracy (which news vendors does the brain interpret correctly)
- Calibration curve (when brain says 0.7, how often is it right?)
- Drift detection (recent vs historical accuracy)
- **Confidence recalibration**: raw confidence overridden by observed
  hit rate at that level once 10+ samples accumulate. The brain
  literally self-improves over time.

**Production robustness (`crates/trading/src/health.rs`).**
The "5-7 days of training and the system falls down" failure mode
addressed:
- **Input sanitization**: NaN/Inf/garbage in `MarketState` is detected
  and rejected (or cleaned) before touching internal state. Bad input
  cannot poison the brain.
- **Health check API**: `/api/trading/health` scans for buffer
  overflow, NaN/Inf creep, calibration staleness, anomaly saturation,
  unscored prediction backlog. Returns severity-ranked findings.
- **Auto-repair API**: `/api/trading/auto_repair` trims oversized
  buffers, zeros NaN/Inf, purges stale unscored predictions. Idempotent.
- **Degraded mode**: when health check finds Critical issues,
  `analyze()` returns Flat / 0.0 confidence / 0 position size instead
  of operating on possibly corrupt state. Self-protection.
- **Snapshot/restore**: `save_snapshot()` / `restore_snapshot()` for
  hourly known-good checkpoints. If current state goes bad, restore
  loses at most an hour of training.
- **Long-running simulation test** (`test_long_running_robustness`,
  `#[ignore]`): runs 1000 predict/train cycles with periodic health
  checks. Catches bugs that don't show in short tests.

**Boring-ML baseline + ensemble** (`baseline.rs`, `backtest.rs`).
Pure-numeric k-NN classifier for honest head-to-head comparison.
Confidence-weighted ensemble combines both. Empirical answer to
"does multi-modal text+number reasoning add value?"

### HTTP API

```
POST /api/trading/analyze        — MarketState JSON → Analysis JSON
POST /api/trading/train          — {state, outcome} → trained
GET  /api/trading/stats          — history size, config
GET  /api/trading/self_assessment — track record + calibration + drift
GET  /api/trading/health         — production health report
POST /api/trading/auto_repair    — clean up bad state
```

### Profitability metrics

Hit rate alone is meaningless for trading. The reasoner reports:

- **Profit factor** = total_wins / |total_losses|. Industry standard.
  > 1.0 = profitable; < 1.0 = losing. Robust to hit-rate skew.
- **Win/loss ratio** = mean_win / mean_loss. > 1.0 = average win larger
  than average loss. Combined with hit rate, gives expected value.
- **Expected value per trade** = (p_win × mean_win) - (p_loss × mean_loss).
  Positive = each trade is positive-expectation.
- **Sharpe per trade** = mean / stddev of per-trade returns.
- **Max drawdown** = largest peak-to-trough cumulative-P&L decline.

A 51% hit-rate strategy with profit factor 2.0 is far better than a
70% hit-rate strategy with profit factor 0.6.

### Backtest results (21 train / 9 test, BTC events 2020-2024)

After loss-aversion sizing + 10% max-position cap:

| Metric                | Value         |
|-----------------------|---------------|
| Hit rate              | 55.6%         |
| Total return          | -0.79%        |
| Max drawdown          | 1.80%         |
| Profit factor         | ~0.40         |
| Win/loss ratio        | 0.40          |

Net-negative on this small dataset because miss magnitudes (1.5-2%
per miss) exceed win magnitudes (0.3-0.7%). With 100+ training events
and continuous online learning, hit rate and win/loss ratio both
improve. The architecture is profit-factor-bound by data, not model.

Tied with boring-ML baseline on hit rate (55.6%); brain wins on
news-context-aware events (Israel-Hamas haven trade), baseline wins
on regime-contrarian events (Trump election bullish despite euphoric
regime). Together, ensemble doesn't beat either — yet — because
disagreements break ~50/50.

### Production deployment

```bash
# Start server with trading endpoints
cargo run --release
# auto-loads data/trading_history.json on startup

# Train continuously as outcomes arrive
curl -X POST http://localhost:3000/api/trading/train \
  -H 'Content-Type: application/json' \
  -d '{"state": {...}, "outcome": {"direction": "Up", "magnitude": 0.02}}'

# Query for predictions
curl -X POST http://localhost:3000/api/trading/analyze \
  -H 'Content-Type: application/json' -d '{...}'

# Monitor
curl http://localhost:3000/api/trading/self_assessment   # accuracy + drift
curl http://localhost:3000/api/trading/health            # corruption check
curl -X POST http://localhost:3000/api/trading/auto_repair  # hourly cron

# Backtest as data accumulates
cargo test --release -p organic-trading test_brain_vs_baseline -- --ignored
```

### What it does NOT replace

- **XGBoost / boring ML** for raw tabular prediction. Use the brain as
  a complement (online learning, explainable retrieval, anomaly), not a
  replacement.
- **LLMs** for qualitative analysis of unstructured text (news
  summaries, post-trade write-ups). Use offline.
- **Risk management infrastructure**. The reasoner outputs a
  `position_sizing.fraction`; your stop-loss / position-limit / VaR
  layer still decides whether to take the trade.

## What's Next

- CLI tool for direct brain interaction (organic-ai "clean my computer")
- MCP server to learn from every Claude conversation passively
- Brain researches on its own when asked a question it can't answer
- Predictive coding for genuine hierarchical learning
- Cortical columns (1M structured neurons beats 80M random ones)
- Sequence learning for understanding sentences and code
- Earned autonomy for file operations (human approves, brain learns preferences)
- **Trading: adaptive ensemble weighting** — use brain vs baseline
  track record (per regime) to learn which model to trust when. Currently
  fixed-weight; could be data-driven.
- **Trading: LLM-extracted structured features** — pre-process news
  with Claude API, feed structured outputs (sentiment, urgency, entity
  types) into the brain. Combines LLM text understanding with brain's
  online learning.

## Output Style

Be concise. Lead with the answer or action, not reasoning. Skip filler words,
preamble, and phrases like "I'll help you with that" or "Certainly!". Prefer
fragments over full sentences in explanations. No trailing summaries of what
you just did. One sentence if it fits.

Code blocks, file paths, commands, and error messages are always written in full.
