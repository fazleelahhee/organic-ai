//! Position lifecycle: open, hold, exit, optionally re-enter.
//!
//! Without this module, the trading reasoner only handles ENTRY
//! decisions — it tells you "go long here" but has no opinion on when
//! to close. For day / swing / range-trading strategies, the EXIT and
//! RE-ENTRY decisions are at least as important as the entry.
//!
//! This module adds:
//!
//! - `OpenPosition`: tracks an active trade — entry state, direction,
//!   size, configured TP/SL, max-favorable-excursion for trailing stops.
//!
//! - `ExitReason` enum: TakeProfit, StopLoss, TrailingStop, Reverse,
//!   Timeout, Hold (don't exit). Captures *why* the system closed,
//!   so a downstream P&L analyzer can attribute outcomes to the right
//!   decision rule.
//!
//! - `should_exit()` on TradingBrain: takes an OpenPosition + current
//!   MarketState, returns an ExitDecision. The brain combines fixed
//!   TP/SL gates with brain-driven gates (regime change, EV-no-longer-
//!   positive, opposite direction with high confidence).
//!
//! - `should_reentry()`: after a position closes, decide whether to
//!   immediately re-open in the brain's currently-predicted direction.
//!   Range-trading strategy = aggressive re-entry; trend-following =
//!   wait for fresh signal.
//!
//! ## Design choices
//!
//! 1. **Take-profit and stop-loss are CONFIGURABLE per position**, not
//!    global. Different regimes warrant different targets — euphoric
//!    bull might use TP=8%/SL=2%, choppy ranges might use TP=3%/SL=1%.
//!    The brain can suggest TP/SL based on similar past patterns, but
//!    the position carries its own settings.
//!
//! 2. **Trailing stop is opt-in**. When set, the position records the
//!    max-favorable price move and triggers exit if price gives back
//!    `trailing_stop_pct` of that gain. Lets winners run while
//!    protecting profits.
//!
//! 3. **Brain-driven exits override fixed gates only when confidence
//!    is high**. We don't want jittery brain reads to flip in and out
//!    of positions on marginal signal — that's how slippage eats edge.
//!
//! 4. **Fixed gates take priority over brain gates**. If price hits
//!    take-profit, exit even if the brain still says hold. Discipline
//!    over conviction, especially for risk management.

use crate::{Direction, MarketState, TradingBrain};
use serde::{Deserialize, Serialize};

/// A currently-open trading position. Carries everything needed to
/// decide its fate: entry conditions, direction, size, configured
/// risk parameters, and tracked metadata for trailing-stop logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenPosition {
    /// Encoded state key at entry — for self-assessment matching of
    /// the realized outcome back to the prediction that opened this.
    pub entry_state_key: String,
    /// Price at entry. P&L is computed against this.
    pub entry_price: f64,
    /// Long (Up) or short (Down). Flat = no trade — should never appear here.
    pub direction: Direction,
    /// Size as fraction of capital, in [0, max_position_fraction].
    /// Always positive (sign is in `direction`).
    pub size: f32,
    /// Take-profit threshold as fraction. 0.04 = close at +4% favorable
    /// move from entry. Hard gate — exits even if brain says hold.
    pub take_profit_pct: f32,
    /// Stop-loss threshold as fraction. 0.02 = close at -2% adverse
    /// move from entry. Hard gate — protects against catastrophic loss.
    pub stop_loss_pct: f32,
    /// Optional trailing-stop fraction. When set, position tracks the
    /// max-favorable price move and closes if price gives back this
    /// fraction of the gain. Lets winners run.
    pub trailing_stop_pct: Option<f32>,
    /// Tick (or wall-clock seconds) at which position was opened. Used
    /// for the timeout exit gate.
    pub opened_at: u64,
    /// Maximum hold duration before forced exit. None = no timeout.
    pub max_hold_ticks: Option<u64>,
    /// Best price seen since opening. Used for trailing-stop logic.
    /// For longs: max price seen. For shorts: min price seen.
    /// Updated by `update_market()` on every new tick.
    pub max_favorable_price: f64,
}

impl OpenPosition {
    /// Construct a new open position. Common defaults: TP=4%, SL=2%,
    /// no trailing stop, no timeout. Caller can override per-position.
    pub fn open(
        entry_state_key: String,
        entry_price: f64,
        direction: Direction,
        size: f32,
        opened_at: u64,
    ) -> Self {
        Self {
            entry_state_key,
            entry_price,
            direction,
            size,
            take_profit_pct: 0.04,
            stop_loss_pct: 0.02,
            trailing_stop_pct: None,
            opened_at,
            max_hold_ticks: None,
            max_favorable_price: entry_price,
        }
    }

    /// Builder: set take-profit threshold (e.g. 0.04 for +4%).
    pub fn with_tp(mut self, tp: f32) -> Self { self.take_profit_pct = tp; self }
    /// Builder: set stop-loss threshold (e.g. 0.02 for -2%).
    pub fn with_sl(mut self, sl: f32) -> Self { self.stop_loss_pct = sl; self }
    /// Builder: enable trailing stop (e.g. 0.015 for "give back 1.5%").
    pub fn with_trailing_stop(mut self, t: f32) -> Self {
        self.trailing_stop_pct = Some(t); self
    }
    /// Builder: set timeout duration in ticks (or seconds).
    pub fn with_timeout(mut self, ticks: u64) -> Self {
        self.max_hold_ticks = Some(ticks); self
    }

    /// Update the max-favorable-price tracker with a new market price.
    /// Call this on every new state for accurate trailing-stop logic.
    pub fn update_market(&mut self, current_price: f64) {
        match self.direction {
            Direction::Up => {
                if current_price > self.max_favorable_price {
                    self.max_favorable_price = current_price;
                }
            }
            Direction::Down => {
                if current_price < self.max_favorable_price {
                    self.max_favorable_price = current_price;
                }
            }
            Direction::Flat => {} // shouldn't happen; ignore
        }
    }

    /// Signed P&L percentage at the given current price. Positive = winning,
    /// negative = losing. Sign accounts for position direction (long vs short).
    pub fn pnl_pct(&self, current_price: f64) -> f64 {
        let move_pct = (current_price - self.entry_price) / self.entry_price;
        match self.direction {
            Direction::Up => move_pct,
            Direction::Down => -move_pct,
            Direction::Flat => 0.0,
        }
    }

    /// Realized P&L in capital fraction = size × pnl_pct.
    pub fn realized_pnl(&self, current_price: f64) -> f64 {
        self.size as f64 * self.pnl_pct(current_price)
    }
}

/// Why an exit decision was made.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExitReason {
    /// Don't exit — keep holding.
    Hold,
    /// Hit configured take-profit threshold (favorable move).
    TakeProfit,
    /// Hit configured stop-loss threshold (adverse move).
    StopLoss,
    /// Trailing stop triggered — gave back from peak.
    TrailingStop,
    /// Brain now predicts opposite direction with high confidence.
    Reverse,
    /// EV gate flipped negative — pattern no longer profitable to hold.
    EvNegative,
    /// Held too long without TP/SL hit.
    Timeout,
}

/// Output of `should_exit()`. The reason is the primary signal; the
/// realized P&L is the dollar (or fraction-of-capital) outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitDecision {
    pub reason: ExitReason,
    pub realized_pnl: f64,
    pub current_pct_move: f64,
}

impl ExitDecision {
    pub fn should_close(&self) -> bool {
        self.reason != ExitReason::Hold
    }
}

impl TradingBrain {
    /// Decide whether to close an open position given the current
    /// market state. Combines fixed TP/SL gates with brain-driven
    /// gates (regime change, EV flip).
    ///
    /// Priority order (first matching wins):
    ///   1. StopLoss   (hard floor — risk management)
    ///   2. TakeProfit (hard ceiling — discipline)
    ///   3. Timeout    (held too long)
    ///   4. TrailingStop (give back from peak)
    ///   5. Reverse    (brain confidently says opposite direction)
    ///   6. EvNegative (brain's EV gate now negative)
    ///   7. Hold       (default: keep holding)
    pub fn should_exit(
        &mut self,
        position: &OpenPosition,
        current: &MarketState,
        current_tick: u64,
    ) -> ExitDecision {
        let pct_move = position.pnl_pct(current.price);
        let pnl = position.size as f64 * pct_move;

        // 1. Stop-loss: hard adverse-move gate. Highest priority —
        //    overrides everything including brain conviction.
        if pct_move <= -(position.stop_loss_pct as f64) {
            return ExitDecision {
                reason: ExitReason::StopLoss, realized_pnl: pnl,
                current_pct_move: pct_move,
            };
        }

        // 2. Take-profit: hard favorable-move gate. Discipline beats
        //    "I think it has more room to run." If brain wants to
        //    re-enter after TP, that's a separate decision.
        if pct_move >= position.take_profit_pct as f64 {
            return ExitDecision {
                reason: ExitReason::TakeProfit, realized_pnl: pnl,
                current_pct_move: pct_move,
            };
        }

        // 3. Timeout: position held too long. Stale capital is dead
        //    capital — close and look for a fresher signal.
        if let Some(max) = position.max_hold_ticks {
            if current_tick.saturating_sub(position.opened_at) >= max {
                return ExitDecision {
                    reason: ExitReason::Timeout, realized_pnl: pnl,
                    current_pct_move: pct_move,
                };
            }
        }

        // 4. Trailing stop: position has had favorable move but is now
        //    giving back. Closes at the configured giveback threshold.
        if let Some(trail_pct) = position.trailing_stop_pct {
            let giveback = match position.direction {
                Direction::Up => {
                    (position.max_favorable_price - current.price) / position.max_favorable_price
                }
                Direction::Down => {
                    (current.price - position.max_favorable_price) / position.max_favorable_price
                }
                Direction::Flat => 0.0,
            };
            // Only triggers if we've actually had favorable movement first.
            let had_favorable = match position.direction {
                Direction::Up => position.max_favorable_price > position.entry_price,
                Direction::Down => position.max_favorable_price < position.entry_price,
                Direction::Flat => false,
            };
            if had_favorable && giveback >= trail_pct as f64 {
                return ExitDecision {
                    reason: ExitReason::TrailingStop, realized_pnl: pnl,
                    current_pct_move: pct_move,
                };
            }
        }

        // 5/6. Brain-driven gates. Run analyze() to get current view.
        // Only override the "hold" default when the brain is
        // confidently against the position, not on marginal signal —
        // jittery exits eat edge to slippage.
        let analysis = self.analyze(current);

        // 5. Reverse: brain now says opposite direction with high
        //    confidence AND meaningful position size. This is a
        //    real regime-change signal.
        if analysis.direction != position.direction
            && analysis.direction != Direction::Flat
            && analysis.confidence > 0.65
            && analysis.position_sizing.fraction.abs() > 0.05
        {
            return ExitDecision {
                reason: ExitReason::Reverse, realized_pnl: pnl,
                current_pct_move: pct_move,
            };
        }

        // 6. EV-negative: brain's EV gate now skips this kind of trade.
        //    Edge has eroded — close at current and look elsewhere.
        if analysis.position_sizing.fraction == 0.0
            && analysis.direction == position.direction
        {
            return ExitDecision {
                reason: ExitReason::EvNegative, realized_pnl: pnl,
                current_pct_move: pct_move,
            };
        }

        // 7. Default: hold.
        ExitDecision {
            reason: ExitReason::Hold, realized_pnl: pnl,
            current_pct_move: pct_move,
        }
    }

    /// After closing a position, decide whether to immediately open a
    /// new one in the brain's currently-predicted direction. Used for
    /// range-trading / cycling strategies where the brain says
    /// "uptrend continues" after a TP exit.
    ///
    /// Returns Some(direction, size) if re-entry is warranted, None to
    /// stay flat. Re-entry requires:
    ///   - Brain confidence > 0.5
    ///   - Position size > 3% (meaningful conviction)
    ///   - Anomaly < 0.85 (not in highly-novel territory)
    pub fn should_reentry(&mut self, current: &MarketState) -> Option<(Direction, f32)> {
        let analysis = self.analyze(current);
        if analysis.confidence > 0.5
            && analysis.position_sizing.fraction.abs() > 0.03
            && analysis.anomaly_score < 0.85
            && analysis.direction != Direction::Flat
        {
            Some((analysis.direction, analysis.position_sizing.fraction.abs()))
        } else {
            None
        }
    }

    /// Suggest a take-profit and stop-loss for a new position based on
    /// the magnitudes of similar past patterns. Returns (tp_pct, sl_pct).
    /// Falls back to defaults (4% TP, 2% SL) if the brain has no
    /// magnitude signal.
    ///
    /// Logic:
    ///   - TP = mean_win_magnitude × 1.0 (close near where similar
    ///     past wins finished — don't overstay)
    ///   - SL = mean_loss_magnitude × 0.7 (tighter than typical loss
    ///     so we exit before catastrophic moves)
    ///   - Both clamped to reasonable ranges (TP: 1%-15%, SL: 0.5%-5%)
    pub fn suggest_targets(&mut self, state: &MarketState) -> (f32, f32) {
        let analysis = self.analyze(state);
        let dist = &analysis.outcome_distribution;
        let tp = if dist.mean_win_magnitude > 1e-6 {
            (dist.mean_win_magnitude * 1.0).clamp(0.01, 0.15)
        } else {
            0.04
        };
        let sl = if dist.mean_loss_magnitude > 1e-6 {
            (dist.mean_loss_magnitude * 0.7).clamp(0.005, 0.05)
        } else {
            0.02
        };
        (tp, sl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NewsItem, TimeContext, Outcome};

    fn s(price: f64) -> MarketState {
        MarketState {
            price, volume: 1000.0,
            recent_return: 0.01, volatility: 0.02,
            indicators: vec![("rsi".into(), 55.0)],
            news: vec![],
            timestamp: Some(TimeContext { hour_utc: 14, day_of_week: 2 }),
        }
    }

    fn open_long(entry_price: f64) -> OpenPosition {
        OpenPosition::open(
            "test_key".to_string(), entry_price, Direction::Up, 0.10, 0,
        )
    }

    #[test]
    fn test_pnl_pct_long() {
        let pos = open_long(100.0);
        // Up 5% → +5% PnL on a long
        assert!((pos.pnl_pct(105.0) - 0.05).abs() < 1e-6);
        // Down 3% → -3% PnL on a long
        assert!((pos.pnl_pct(97.0) - (-0.03)).abs() < 1e-6);
    }

    #[test]
    fn test_pnl_pct_short() {
        let pos = OpenPosition::open(
            "k".into(), 100.0, Direction::Down, 0.10, 0);
        // Up 5% → -5% PnL on a short
        assert!((pos.pnl_pct(105.0) - (-0.05)).abs() < 1e-6);
        // Down 3% → +3% PnL on a short
        assert!((pos.pnl_pct(97.0) - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_take_profit_triggers() {
        let mut tb = TradingBrain::new_small(2048);
        let pos = open_long(100.0).with_tp(0.04);
        let exit = tb.should_exit(&pos, &s(105.0), 1);
        assert_eq!(exit.reason, ExitReason::TakeProfit);
        assert!(exit.realized_pnl > 0.0);
    }

    #[test]
    fn test_stop_loss_triggers() {
        let mut tb = TradingBrain::new_small(2048);
        let pos = open_long(100.0).with_sl(0.02);
        let exit = tb.should_exit(&pos, &s(97.0), 1);
        assert_eq!(exit.reason, ExitReason::StopLoss);
        assert!(exit.realized_pnl < 0.0);
    }

    #[test]
    fn test_stop_loss_priority_over_take_profit() {
        // SL hit AND TP hit at the same moment (silly setup but tests
        // priority): SL should win.
        let mut tb = TradingBrain::new_small(2048);
        let pos = open_long(100.0).with_tp(0.001).with_sl(0.001);
        // Move down 2% — both thresholds crossed in opposite directions
        // (TP=+0.1% not crossed, SL=-0.1% crossed). Test really
        // verifies SL fires when applicable.
        let exit = tb.should_exit(&pos, &s(98.0), 1);
        assert_eq!(exit.reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_hold_when_no_threshold_hit() {
        let mut tb = TradingBrain::new_small(2048);
        let pos = open_long(100.0).with_tp(0.05).with_sl(0.05);
        // +1% move: between TP and SL, no exit triggers.
        let exit = tb.should_exit(&pos, &s(101.0), 1);
        // Untrained brain probably won't reverse-signal, so we get Hold
        // OR EvNegative (zero-position-size signal). Both are non-TP,
        // non-SL, non-Reverse — we just verify it isn't TP/SL.
        assert_ne!(exit.reason, ExitReason::TakeProfit);
        assert_ne!(exit.reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_trailing_stop_tracks_max_favorable() {
        let mut pos = open_long(100.0).with_trailing_stop(0.02);
        pos.update_market(105.0);
        assert_eq!(pos.max_favorable_price, 105.0);
        pos.update_market(103.0); // pull back
        assert_eq!(pos.max_favorable_price, 105.0); // doesn't decrease
        pos.update_market(110.0);
        assert_eq!(pos.max_favorable_price, 110.0); // new high
    }

    #[test]
    fn test_trailing_stop_triggers_on_giveback() {
        let mut tb = TradingBrain::new_small(2048);
        // Use wide TP so the trailing stop has a chance to fire first.
        // Without the wide TP, TakeProfit (4% default) hits before
        // any trailing-stop logic can activate.
        let mut pos = open_long(100.0)
            .with_tp(0.20)              // very wide TP, won't trigger
            .with_sl(0.10)              // wide SL, won't trigger
            .with_trailing_stop(0.02);
        // Run up to 110 (+10% favorable), then back to 107.5
        // (giveback from peak: (110-107.5)/110 = 2.27% > 2% threshold).
        pos.update_market(110.0);
        let exit = tb.should_exit(&pos, &s(107.5), 1);
        assert_eq!(exit.reason, ExitReason::TrailingStop);
    }

    #[test]
    fn test_trailing_stop_no_trigger_without_favorable_first() {
        let mut tb = TradingBrain::new_small(2048);
        let pos = open_long(100.0).with_trailing_stop(0.02);
        // Price never went above entry — trailing stop shouldn't fire
        // even if current is below entry. (StopLoss might fire instead;
        // here we use generous SL to isolate trailing behavior.)
        let pos = pos.with_sl(0.10);
        let exit = tb.should_exit(&pos, &s(99.0), 1);
        // Not TrailingStop — that requires prior favorable move.
        assert_ne!(exit.reason, ExitReason::TrailingStop);
    }

    #[test]
    fn test_timeout_triggers() {
        let mut tb = TradingBrain::new_small(2048);
        let pos = open_long(100.0)
            .with_tp(0.10).with_sl(0.10)  // wide enough to not trigger
            .with_timeout(100);
        let exit = tb.should_exit(&pos, &s(101.0), 200); // 200 ticks > 100 timeout
        assert_eq!(exit.reason, ExitReason::Timeout);
    }

    #[test]
    fn test_realized_pnl_signs() {
        // Long up 5% with 10% size → +0.005 capital gain
        let pos = open_long(100.0);
        assert!((pos.realized_pnl(105.0) - 0.005).abs() < 1e-6);
        // Long down 3% with 10% size → -0.003 capital loss
        assert!((pos.realized_pnl(97.0) - (-0.003)).abs() < 1e-6);
    }

    /// Suggest-targets uses similar-past-pattern magnitudes when
    /// available, falls back to defaults when not. Verify both paths.
    #[test]
    fn test_suggest_targets_falls_back_to_defaults() {
        let mut tb = TradingBrain::new_small(2048);
        // Untrained brain has no history — should return defaults.
        let (tp, sl) = tb.suggest_targets(&s(100.0));
        assert_eq!(tp, 0.04);
        assert_eq!(sl, 0.02);
    }

    #[test]
    fn test_suggest_targets_uses_history_when_available() {
        let mut tb = TradingBrain::new_small(2048);
        let state = s(100.0);
        // Train a strong Up pattern: similar wins are 6%.
        for _ in 0..15 {
            tb.train_on_outcome(&state, &Outcome::new(Direction::Up, 0.06));
        }
        // Train some Down "losses" (when predicting Up) at 3%.
        for _ in 0..5 {
            tb.train_on_outcome(&state, &Outcome::new(Direction::Down, 0.03));
        }
        let (tp, sl) = tb.suggest_targets(&state);
        // TP should reflect mean win magnitude (~6%, clamped to 15%).
        // SL should be 0.7 × mean loss (~2.1%, clamped to 5%).
        assert!(tp > 0.04, "TP should pick up on 6% mean win, got {}", tp);
        assert!(sl < 0.05, "SL should be tighter than 5% cap, got {}", sl);
    }
}
