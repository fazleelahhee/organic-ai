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
use organic_trading::self_assessment::SelfAssessment;
use organic_trading::health::HealthReport;
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

/// Self-assessment endpoint — overall + per-regime + per-source
/// accuracy, calibration curve, drift flag. The brain's track record,
/// surfaced for the human / risk-management system to inspect.
/// Production trading should poll this periodically.
pub async fn self_assessment(
    Extension(tb): Extension<TradingState>,
) -> Json<SelfAssessment> {
    let tb = tb.lock().await;
    Json(tb.self_assessment())
}

/// Production health check. Scans for the long-running failure modes:
/// NaN/Inf, oversized buffers, stale calibration, anomaly saturation,
/// excessive unscored prediction backlog. Returns a HealthReport with
/// per-finding severity (Info/Warning/Critical). Side-effect-free.
///
/// Recommended polling: every 5-15 minutes. Alert any Critical
/// immediately.
pub async fn health(
    Extension(tb): Extension<TradingState>,
) -> Json<HealthReport> {
    let tb = tb.lock().await;
    Json(tb.health_check())
}

/// Auto-repair handler. Trims oversized buffers, zeros NaN/Inf,
/// purges stale unscored predictions. Returns the post-repair health
/// report. Idempotent, but call sparingly — clean-up has cost and
/// shouldn't run on every request.
///
/// Suggested cadence: hourly cron, or after a Warning-level health
/// finding fires.
pub async fn auto_repair(
    Extension(tb): Extension<TradingState>,
) -> Json<HealthReport> {
    let mut tb = tb.lock().await;
    Json(tb.auto_repair())
}
