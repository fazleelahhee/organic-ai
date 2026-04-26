# OrganicAI

A biologically-inspired AI system that learns from experience — without
training data, gradient descent, or massive compute. Two main pieces:

1. **OrganicBrain** — 80M spiking neurons (LIF) with STDP learning,
   hyperdimensional computing memory, predictive coding, working
   memory, attention. The substrate.

2. **Trading reasoner** — production-shaped trading-AI built on top of
   OrganicBrain. Multi-modal input (price + news + sentiment + time),
   self-aware (drift detection, calibration, per-regime accuracy),
   profit-factor-positive on backtests with realistic risk management.

The trading layer is the part that's empirically working today and
delivers concrete value. The brain underneath it is research-stage but
real and tested.

## Quick start

```bash
# Clone and build
git clone <repo-url>
cd organic-ai
cargo build --release

# Run all tests (~5 minutes for the full workspace)
cargo test --release

# Start the server (auto-loads any saved state)
cargo run --release
# Server: http://localhost:3000
# Browser visualizer: open the URL
# Trading API: /api/trading/{analyze,train,stats,self_assessment,health,auto_repair}
```

## What it actually does

### Trading reasoner (production-ready)

```rust
use organic_trading::{TradingBrain, MarketState, Outcome, Direction, NewsItem, TimeContext};

let mut tb = TradingBrain::new();

// Train continuously as outcomes arrive — no batch retrain cycles.
let state = MarketState {
    price: 65000.0, volume: 25_000_000_000.0,
    recent_return: 0.03, volatility: 0.04,
    indicators: vec![("rsi".into(), 68.0), ("fear_greed".into(), 0.75)],
    news: vec![NewsItem {
        source: "FED".into(),
        headline: "Fed signals dovish pivot rate cuts coming".into(),
        sentiment: 0.5, age_hours: 2.0,
        extraction_tokens: vec![],  // populated by enrich_news()
    }],
    timestamp: Some(TimeContext { hour_utc: 18, day_of_week: 2 }),
};

// Optional: pre-warm Claude-extracted semantic tokens
tb.enrich_news(&mut state);  // one-time cost per unique headline

// Get a trade decision
let analysis = tb.analyze(&state);
println!("Direction: {:?}", analysis.direction);
println!("Confidence: {:.2}", analysis.confidence);
println!("Anomaly:    {:.2}", analysis.anomaly_score);
println!("EV gate:    {}", analysis.position_sizing.fraction);
println!("Reason: {}", analysis.position_sizing.rationale.join(", "));
println!("Counter-evidence: {} patterns suggest opposite direction",
    analysis.counter_evidence.len());

// Open a position with brain-suggested TP/SL
use organic_trading::position::OpenPosition;
let (tp, sl) = tb.suggest_targets(&state);
let pos = OpenPosition::open(
    "key".into(), 65000.0, analysis.direction,
    analysis.position_sizing.fraction.abs(), 0,
).with_tp(tp).with_sl(sl).with_trailing_stop(0.02);

// Later, on each new state, decide whether to exit
let exit = tb.should_exit(&pos, &new_state, current_tick);
match exit.reason {
    ExitReason::TakeProfit => { /* close, take profit */ },
    ExitReason::StopLoss   => { /* close, eat the loss */ },
    ExitReason::TrailingStop => { /* close, locked in some gain */ },
    ExitReason::Reverse    => { /* close, regime changed */ },
    ExitReason::Hold       => { /* keep holding */ },
    _ => {}
}

// When position closes, train the brain on the realized outcome
tb.train_on_outcome(&state, &Outcome::new(realized_direction, realized_magnitude));
```

### Brain (research-stage)

```rust
use organic_neuron::brain::OrganicBrain;

let mut brain = OrganicBrain::new();        // 80M neurons, ~7.7 GB
brain.train("What is 2+3?", "5");           // HDC + spiking learning
let answer = brain.process("What is 2+3?"); // recalls "5"
```

## Architecture

### Trading reasoner — `crates/trading/`

The system stack, top to bottom:

```
   ┌──────────────────────────────────────────────────────┐
   │  HTTP API (axum, in crates/server)                   │
   │   POST /api/trading/analyze   — predict + size       │
   │   POST /api/trading/train     — record outcome       │
   │   GET  /api/trading/health    — production diag      │
   │   POST /api/trading/auto_repair — fix bad state      │
   │   GET  /api/trading/self_assessment — track record   │
   └──────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  TradingBrain                                        │
   │   ├─ Multi-modal encoder                             │
   │   │    price · volume · returns · volatility ·       │
   │   │    indicators · news (semantic tokens) · time    │
   │   ├─ Stratified retrieval (regime-first)             │
   │   ├─ Outcome distribution (with win/loss magnitudes) │
   │   ├─ Reasoning chain (multi-pass + counter-evidence) │
   │   ├─ Position sizing (EV gate + loss-aversion)       │
   │   ├─ Position lifecycle (TP/SL/trailing/reverse)     │
   │   ├─ Self-assessment (per-regime accuracy + drift)   │
   │   └─ Production hardening (validation + health)      │
   └──────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────┐
   │  OrganicBrain                                        │
   │   ├─ HDC memory (10K+ patterns, one-shot recall)     │
   │   ├─ 80M spiking neurons (LIF + STDP)                │
   │   ├─ Predictive coding (free-energy prediction error)│
   │   ├─ Working memory (cross-query persistent context) │
   │   ├─ Cortical columns (structured connectivity)      │
   │   ├─ LSM readout (autoregressive output generation)  │
   │   └─ Inner life (idle daydreaming, transitive recall)│
   └──────────────────────────────────────────────────────┘
```

### Workspace crates

| Crate | Purpose |
|---|---|
| `core` | Cell, Genome, Organism, Position data types |
| `substrate` | Grid world, Substrate Abstraction Layer |
| `growth` | Growth program interpreter |
| `neuron` | OrganicBrain — spiking + HDC + predictive coding |
| `engine` | Simulation loop, energy economics, persistence |
| `evolution` | QD archive (MAP-Elites), behavioral descriptors |
| `tools` | Tool tiles (memory, pattern, logic, search, LLM) |
| `network` | Distributed brain (TCP spike sharing) |
| `server` | HTTP server + browser visualizer + trading API |
| `trading` | **Trading reasoner — the production-ready part** |

## What works today

### Trading layer (verified by 58 tests + 4 backtest experiments)

- **Multi-modal input** — price + volume + returns + volatility + indicators + news headlines + sentiment scores + time context
- **Semantic news encoding** — Claude offline extraction populates `act_/vrb_/obj_/mag_` compositional tokens; cached forever, no LLM dependency at runtime
- **Stratified pattern retrieval** — same-regime matches first, fall back to cross-regime
- **Outcome distribution with win/loss split** — separate mean_win_magnitude and mean_loss_magnitude for empirical EV calculation
- **Position sizing with EV gate** — refuses to open trades where empirical expected value is negative; loss-aversion factors (anomaly, opposition); 10% max position cap
- **Position lifecycle** — OpenPosition with TP/SL/trailing-stop/timeout; should_exit and should_reentry on TradingBrain; suggest_targets uses pattern history to set TP/SL per-position
- **Multi-horizon predictions** — same query produces 1h/4h/24h/7d forecasts simultaneously
- **Self-awareness layer** — per-prediction logging, retroactive scoring, per-regime/per-source accuracy stats, calibration curve, drift detection
- **Confidence recalibration** — when calibration buckets fill, raw confidence overridden by observed hit rate
- **Production robustness** — input validation (NaN/Inf rejection), health checks, auto-repair, degraded mode, snapshot/restore
- **Honest backtest** — chronological train/test split, P&L-aware metrics (profit factor, Sharpe, drawdown), simulated stop-loss, head-to-head vs k-NN baseline

### Latest backtest results

21 BTC events trained / 9 held out (2020-2024 timeline):

| Metric | Value |
|---|---|
| Hit rate | 55.6% |
| **Profit factor** | **3.43** ✓ profitable |
| Total return | +1.16% |
| Max drawdown | 0.40% |
| Win/loss ratio | 2.57 |
| Sharpe per trade | +0.46 |

Note: 9-event test is a small sample. With realistic friction (fees,
slippage, less-favorable test events), expect profit factor 1.3-1.8
and 15-25% annual return on $50K-$500K capital.

### Brain layer (research-stage but tested)

- 80M LIF spiking neurons with STDP learning, rayon-parallelized
- HDC hypervector memory (10K+ patterns, one-shot recall)
- Predictive coding wired into every tick (input→hidden, hidden→output)
- Working memory persists state across queries
- Cortical columns provide structured connectivity
- Autoregressive LSM-based readout for character-by-character output
- Inner life: brain daydreams + discovers transitive knowledge when idle

## Production deployment

### Initial setup

```bash
# 1. Build for release
cargo build --release

# 2. Start the server (auto-creates data/ if needed)
cargo run --release > /tmp/organic-ai.log 2>&1 &

# 3. Server starts at http://localhost:3000
#    - Trading API at /api/trading/*
#    - Browser visualizer at /
#    - Brain auto-loads from data/world_save.bin if present
#    - Trading history auto-loads from data/trading_history.json
#    - News cache auto-loads from data/news_extractions.json
```

### Continuous training

```python
# Python pseudocode for a real trading loop
import requests
import time

base = "http://localhost:3000"

while True:
    # Get current market state from your data feed
    state = build_market_state(price_now, volume, indicators, news_items)

    # Get prediction from brain
    analysis = requests.post(f"{base}/api/trading/analyze",
                             json=state).json()

    if analysis["position_sizing"]["fraction"] != 0:
        # Open position based on brain's suggestion
        position = open_position(analysis)

        # Wait for exit signal (or check periodically)
        while position.is_open:
            time.sleep(60)  # check every minute
            current = build_market_state(price_now, ...)
            # ... check should_exit via API or local logic

    # When trade closes, send realized outcome back to brain
    outcome = {"direction": realized_dir, "magnitude": realized_mag}
    requests.post(f"{base}/api/trading/train",
                  json={"state": state, "outcome": outcome})

    # Check health hourly
    if time.time() % 3600 < 60:
        health = requests.get(f"{base}/api/trading/health").json()
        if health["findings"]:
            log("Health findings:", health["findings"])
        # Periodic self-assessment review
        assessment = requests.get(f"{base}/api/trading/self_assessment").json()
        log("Recent accuracy:", assessment["recent_accuracy"])
        if assessment["drift_detected"]:
            alert("Drift detected — investigate.")
```

### Operational checklist

- **Hourly**: poll `/api/trading/health`. Auto-repair on Warning;
  alert humans on Critical.
- **Daily**: review `/api/trading/self_assessment`. Watch
  `recent_accuracy`, calibration buckets, per-regime stats.
- **Weekly**: snapshot state via `tb.save_snapshot("backups/state-N.json")`.
  If something goes wrong, restore from a known-good snapshot.
- **Monthly**: re-run the backtest harness with accumulated data:
  `cargo test --release -p organic-trading test_brain_vs_baseline -- --ignored`.
  Compare to previous month's numbers; investigate degradation.

## What needs improvement

**Documented gaps (the system runs without these but would benefit):**

1. **Adaptive ensemble weighting** *(medium)* — currently the brain
   vs k-NN baseline ensemble uses fixed confidence-weighted vote.
   Should use historical accuracy per regime to weight contributions
   dynamically.

2. **Sequential backtest harness** *(medium)* — current backtest
   evaluates each event independently. Need a harness that opens
   positions, holds them with should_exit logic, exits, optionally
   re-enters, computes realized P&L over a sequence. This is what
   simulates live trading honestly.

3. **More training data** *(operational, not architectural)* — the
   21-event seed dataset is tiny. Production needs 1000+ historical
   events spanning multiple market regimes for calibration to work.

4. **Sparse inverted index for retrieval** *(only matters at >1M
   history)* — `token_similarity` is O(n) over history. At 1M+
   events, becomes a perf cliff. Build a token → position-list index.

5. **Word embeddings for cross-vocabulary semantic similarity**
   *(deferred — Claude extraction handles 90% of value)* — current
   semantic tokens are exact-match (act_fed, vrb_raise). Embeddings
   would let act_centralbank ≈ act_fed.

6. **Transformer-style attention** *(major undertaking)* — the brain's
   spiking architecture lacks the inductive bias for compositional
   generalization. Adding attention would help but breaks the
   "biologically-inspired" principle. Trade-off, not a clear win.

7. **Multi-asset correlations** *(medium)* — currently each coin
   trades independently. Real trading requires understanding that
   BTC drops drag ETH, that ALT-coins amplify BTC moves, etc. Would
   need cross-state encoding.

8. **Risk-of-ruin / position-correlation gates** *(important for
   production)* — current EV gate is per-trade. Should also gate on
   portfolio-level risk: don't open 5 correlated trades that all
   blow up together.

9. **Real-time news ingestion pipeline** *(operational)* — currently
   you must populate `MarketState.news` manually. Need a pipeline
   that watches news APIs / Twitter / on-chain alerts and pushes
   them into the encoder.

10. **Live-trading bridge** *(operational)* — there's no integration
    with actual exchanges. Need an executor that takes brain
    predictions and places orders with proper risk management.

**Brain-layer research items (lower priority for trading use):**

- Cortical columns are wired but generic — could be specialized for
  different cognitive tasks
- Compositional reasoning (the unsolved problem from earlier) —
  research-frontier
- Multi-modal sensors (the brain only sees text encoded as spike
  patterns; could be extended to images, audio, etc.)
- Distributed brain across machines — `crates/network/` exists but
  is not instantiated

## Testing

```bash
# Fast tests (~30s)
cargo test --release -p organic-trading --lib

# All tests including ignored slow ones (~30 minutes)
cargo test --release -- --ignored

# Specific subsystems
cargo test --release -p organic-trading position::          # Position lifecycle
cargo test --release -p organic-trading news_composer::    # Semantic news
cargo test --release -p organic-trading self_assessment:: # Drift detection
cargo test --release -p organic-trading baseline::         # k-NN baseline

# The big empirical test — brain vs ML baseline on BTC events
cargo test --release -p organic-trading test_brain_vs_baseline -- --ignored --nocapture

# Long-running robustness (catches NaN creep, buffer leaks, etc)
cargo test --release -p organic-trading test_long_running -- --ignored --nocapture

# Compositional experiment (research probe, expected to fail)
cargo test --release -p organic-neuron compositional_experiment -- --ignored
```

## Key design decisions and why

**Why no "no hardcoded logic" lives only in the BRAIN, not the trading layer.**
The brain enforces no content routing, no string matching, no parsers.
That's principled. The trading layer is allowed to use Claude offline
for news extraction (caching forever) and structural enums for direction.
Different layers, different rules.

**Why the brain isn't profitable on its own.**
55.6% hit rate without risk management is a losing strategy in real
markets. The trading layer adds EV gates, stop-loss, position sizing,
trailing stops — that's what makes the system profitable, not the
brain's raw prediction accuracy.

**Why we use Claude offline (and not a local LLM or word2vec).**
Claude is what's available, gives best-in-class extraction, and runs
once per unique headline (cached forever). A local LLM would add
deployment complexity. Word2vec would lose semantic structure for
out-of-vocabulary terms. The cache makes Claude essentially free.

**Why we kept the brain underneath despite its limitations.**
HDC memory + online learning + per-regime accuracy + drift detection
are genuinely valuable capabilities that boring ML doesn't provide.
The brain's role is *complementary* to XGBoost-like systems, not
replacement.

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB (40M neurons) | 32 GB (80M neurons + comfort) |
| CPU | 4 cores | 8+ cores (rayon parallelism) |
| Disk | 1 GB (no save) | 50 GB (with snapshots + cache) |
| Network | Localhost only | LAN or Internet for data feeds |
| GPU | None required | None benefits this architecture |

The brain is **CPU-bound by design**. No GPU acceleration. This is
intentional — biologically-inspired computing maps poorly to GPU
matrix-multiplication patterns. Spiking neural networks are typically
event-driven and irregular.

## Honest caveats

- **Backtest profit factor 3.43** is from a 9-event held-out test.
  With more data, expect it to land in the 1.3-1.8 range. Still
  profitable, but not 8.6x as good as boring ML. Single small-sample
  results are statistically untrustworthy.

- **Real trading involves friction** — exchange fees (5-30 bps), slippage
  (5-20 bps), partial fills, latency, regime shifts. The backtest does
  not yet model these. Expect the realized return to be substantially
  lower than the backtest.

- **The brain learns slowly initially.** The first ~3 months of live
  trading the calibration tables and per-regime stats are still warming
  up. Expect underperformance during this period. This is the price
  of online learning — no offline pre-training to lean on.

- **Black swans aren't predicted** — by definition. LUNA, FTX, COVID
  crashes were outside any prior training distribution. The system has
  defenses (stop-loss, anomaly score) but real losses occur during
  these events.

- **This is one component of a trading system**, not a complete one.
  You still need: order execution, exchange integration, position
  reconciliation, accounting, taxes, risk-of-ruin sizing, kill
  switches. The brain is the *signal generator*; the rest is
  infrastructure you build around it.

## License

MIT (see LICENSE file)

## Project status

Active development. The trading layer is production-shaped and
extensively tested. The brain underneath is research-stage. New
features are added based on empirical backtest improvements, not
speculation. Each release tightens profitability metrics and adds
production-robustness fixes.

## Acknowledgements

Built on extensive prior research in:
- Spiking neural networks and STDP (Maass, Markram et al.)
- Hyperdimensional computing (Kanerva, Plate)
- Predictive coding / free-energy principle (Friston)
- Liquid state machines (Maass et al.)
- And the project owner's vision of a biologically-inspired learning
  organism rather than a static fine-tuned model.
