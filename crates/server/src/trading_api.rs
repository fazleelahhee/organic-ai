//! HTTP API for the trading reasoner.
//!
//! Three endpoints:
//!   - POST /api/trading/analyze — take a MarketState, return Analysis
//!   - POST /api/trading/train   — train on (state, outcome) pair
//!   - GET  /api/trading/stats   — basic health + history-size info
//!
//! The trading reasoner has its own dedicated brain (separate from the
//! organism-simulation brain that powers /api/message). This separation
//! is intentional: the trading brain's training data is structured
//! market events, not free-text Q&A — mixing them would degrade both.
//!
//! State is wrapped in `Arc<Mutex<TradingBrain>>`. Calls are serialized
//! through the mutex — at production scale this becomes a bottleneck,
//! but for the typical hand-trading-system use case (1-10 queries/sec)
//! it's fine and avoids ownership complexity.

use axum::{Extension, Json, http::StatusCode};
use organic_trading::{TradingBrain, MarketState, Outcome, Analysis};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

pub type TradingState = Arc<Mutex<TradingBrain>>;

#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    pub state: MarketState,
    pub outcome: Outcome,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub history_len: usize,
    pub history_capacity: usize,
    pub reasoning_passes: usize,
}

pub async fn analyze(
    Extension(tb): Extension<TradingState>,
    Json(state): Json<MarketState>,
) -> Result<Json<Analysis>, (StatusCode, String)> {
    let mut tb = tb.lock().await;
    let analysis = tb.analyze(&state);
    Ok(Json(analysis))
}

pub async fn train(
    Extension(tb): Extension<TradingState>,
    Json(req): Json<TrainRequest>,
) -> Result<Json<&'static str>, (StatusCode, String)> {
    let mut tb = tb.lock().await;
    tb.train_on_outcome(&req.state, &req.outcome);
    Ok(Json("trained"))
}

pub async fn stats(
    Extension(tb): Extension<TradingState>,
) -> Json<StatsResponse> {
    let tb = tb.lock().await;
    Json(StatsResponse {
        history_len: tb.history_len(),
        history_capacity: tb.history_capacity,
        reasoning_passes: tb.reasoning_passes,
    })
}
