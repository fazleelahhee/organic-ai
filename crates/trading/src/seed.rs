//! Curated historical Bitcoin events for training and backtesting.
//!
//! Each event is a `HistoricalEvent` containing the market state at the
//! time of a notable news event (Fed announcement, Musk tweet, exchange
//! collapse, geopolitical shock) and the realized 24h-7d price reaction.
//!
//! The events are not exact to the dollar — prices are rounded to
//! representative levels and sentiment scores are estimated from the
//! observed market reaction. The point is to give the trading reasoner
//! a realistic curriculum spanning the full menagerie of crypto-market
//! news shocks: hawkish Fed, dovish Fed, regulatory crackdown,
//! regulatory approval, exchange collapse, exchange resurrection,
//! whale tweet pump, whale tweet dump, macro shock, ETF news.
//!
//! Coverage spans March 2020 through 2024 — five years of realized
//! BTC reactions to news. After training, the brain should learn:
//!   - Fed rate hikes correlate with BTC down (most of the time)
//!   - Spot ETF approval news correlates with BTC up
//!   - Exchange collapses correlate with BTC down
//!   - China regulatory bans correlate with BTC down
//!   - Strong negative sentiment + macro shock = BTC down hard
//!
//! For backtesting: split events into train (chronologically first 70%)
//! and test (remaining 30%) so the brain is evaluated on events it
//! couldn't possibly have memorized.

use crate::{MarketState, Outcome, NewsItem, TimeContext, Direction};

/// One labeled historical event for training or evaluation.
#[derive(Debug, Clone)]
pub struct HistoricalEvent {
    /// Human-readable label describing the event.
    pub label: &'static str,
    /// Approximate date string (YYYY-MM-DD) — informational only.
    pub date: &'static str,
    /// Market state including the news context at the time of the event.
    pub state: MarketState,
    /// Realized direction + magnitude in the 24-hour window after.
    pub outcome: Outcome,
}

fn news(source: &str, headline: &str, sentiment: f64, age_hours: f64) -> NewsItem {
    NewsItem {
        source: source.to_string(),
        headline: headline.to_string(),
        sentiment,
        age_hours,
        extraction_tokens: Vec::new(),
    }
}

fn ts(hour_utc: u8, day_of_week: u8) -> Option<TimeContext> {
    Some(TimeContext { hour_utc, day_of_week })
}

/// Curated BTC event history for training + backtesting. Returns events
/// in chronological order (oldest first), so callers can split
/// chronologically without re-sorting.
pub fn historical_btc_events() -> Vec<HistoricalEvent> {
    vec![
        // ============= 2020: COVID crash, halving, then run-up =============
        HistoricalEvent {
            label: "COVID-19 market crash (Black Thursday)",
            date: "2020-03-12",
            state: MarketState {
                price: 7900.0, volume: 50_000_000_000.0,
                recent_return: -0.30, volatility: 0.18,
                indicators: vec![("rsi".into(), 18.0), ("fear_greed".into(), 0.10)],
                news: vec![
                    news("REUTERS", "WHO declares COVID-19 a global pandemic", -0.85, 6.0),
                    news("MACRO", "Equity markets crash record drops circuit breakers", -0.90, 4.0),
                ],
                timestamp: ts(15, 3),
            },
            outcome: Outcome::new(Direction::Down, 0.40),
        },
        HistoricalEvent {
            label: "Fed slashes rates to zero",
            date: "2020-03-15",
            state: MarketState {
                price: 5300.0, volume: 45_000_000_000.0,
                recent_return: -0.08, volatility: 0.20,
                indicators: vec![("rsi".into(), 25.0)],
                news: vec![
                    news("FED", "Fed cuts rates to zero emergency QE unlimited", 0.40, 2.0),
                    news("MACRO", "Stocks limit down futures crash overnight", -0.60, 8.0),
                ],
                timestamp: ts(22, 6),
            },
            outcome: Outcome::new(Direction::Up, 0.12),
        },
        HistoricalEvent {
            label: "Bitcoin halving",
            date: "2020-05-11",
            state: MarketState {
                price: 8700.0, volume: 35_000_000_000.0,
                recent_return: 0.04, volatility: 0.06,
                indicators: vec![("rsi".into(), 62.0), ("fear_greed".into(), 0.45)],
                news: vec![
                    news("ONCHAIN", "Bitcoin block reward halving completed supply shock narrative", 0.55, 1.0),
                ],
                timestamp: ts(19, 0),
            },
            outcome: Outcome::new(Direction::Up, 0.03),
        },
        HistoricalEvent {
            label: "MicroStrategy first BTC buy",
            date: "2020-08-11",
            state: MarketState {
                price: 11500.0, volume: 28_000_000_000.0,
                recent_return: 0.02, volatility: 0.04,
                indicators: vec![("rsi".into(), 65.0)],
                news: vec![
                    news("CORP", "MicroStrategy purchases bitcoin treasury reserve asset", 0.70, 2.0),
                ],
                timestamp: ts(13, 1),
            },
            outcome: Outcome::new(Direction::Up, 0.05),
        },
        // ============= 2021: First half — Tesla, ATH, Musk pivot =============
        HistoricalEvent {
            label: "Tesla announces $1.5B BTC purchase",
            date: "2021-02-08",
            state: MarketState {
                price: 39000.0, volume: 60_000_000_000.0,
                recent_return: 0.05, volatility: 0.05,
                indicators: vec![("rsi".into(), 70.0), ("fear_greed".into(), 0.78)],
                news: vec![
                    news("TWITTER", "Elon Musk Tesla bought bitcoin SEC filing 1.5 billion", 0.92, 1.5),
                    news("CORP", "Tesla 10-K filing discloses bitcoin treasury position", 0.85, 2.0),
                ],
                timestamp: ts(14, 0),
            },
            outcome: Outcome::new(Direction::Up, 0.18),
        },
        HistoricalEvent {
            label: "Coinbase direct listing",
            date: "2021-04-14",
            state: MarketState {
                price: 63000.0, volume: 75_000_000_000.0,
                recent_return: 0.04, volatility: 0.05,
                indicators: vec![("rsi".into(), 76.0), ("fear_greed".into(), 0.85)],
                news: vec![
                    news("CORP", "Coinbase direct listing Nasdaq COIN public crypto exchange", 0.65, 3.0),
                ],
                timestamp: ts(13, 2),
            },
            outcome: Outcome::new(Direction::Down, 0.04),
        },
        HistoricalEvent {
            label: "Musk suspends Tesla BTC payments",
            date: "2021-05-12",
            state: MarketState {
                price: 49000.0, volume: 70_000_000_000.0,
                recent_return: -0.02, volatility: 0.06,
                indicators: vec![("rsi".into(), 48.0), ("fear_greed".into(), 0.55)],
                news: vec![
                    news("TWITTER", "Elon Musk Tesla suspending bitcoin vehicle purchases environmental", -0.80, 0.5),
                ],
                timestamp: ts(22, 2),
            },
            outcome: Outcome::new(Direction::Down, 0.15),
        },
        HistoricalEvent {
            label: "China mining ban Inner Mongolia escalation",
            date: "2021-05-21",
            state: MarketState {
                price: 37000.0, volume: 95_000_000_000.0,
                recent_return: -0.10, volatility: 0.12,
                indicators: vec![("rsi".into(), 28.0), ("fear_greed".into(), 0.20)],
                news: vec![
                    news("REUTERS", "China State Council bitcoin mining trading crackdown", -0.85, 3.0),
                    news("ONCHAIN", "Hashrate drops China miners migrating Kazakhstan Texas", -0.50, 6.0),
                ],
                timestamp: ts(8, 4),
            },
            outcome: Outcome::new(Direction::Down, 0.20),
        },
        HistoricalEvent {
            label: "El Salvador adopts BTC legal tender",
            date: "2021-06-09",
            state: MarketState {
                price: 33500.0, volume: 50_000_000_000.0,
                recent_return: 0.08, volatility: 0.07,
                indicators: vec![("rsi".into(), 58.0), ("fear_greed".into(), 0.40)],
                news: vec![
                    news("REUTERS", "El Salvador Bukele bitcoin legal tender first nation", 0.75, 4.0),
                ],
                timestamp: ts(16, 2),
            },
            outcome: Outcome::new(Direction::Up, 0.08),
        },
        // ============= 2021: Second half — recovery, ATH =============
        HistoricalEvent {
            label: "Taproot activation announced",
            date: "2021-09-12",
            state: MarketState {
                price: 46000.0, volume: 35_000_000_000.0,
                recent_return: 0.01, volatility: 0.04,
                indicators: vec![("rsi".into(), 60.0)],
                news: vec![
                    news("ONCHAIN", "Bitcoin taproot soft fork activation block height", 0.45, 12.0),
                ],
                timestamp: ts(20, 6),
            },
            outcome: Outcome::new(Direction::Up, 0.04),
        },
        HistoricalEvent {
            label: "China bans all crypto transactions",
            date: "2021-09-24",
            state: MarketState {
                price: 44000.0, volume: 60_000_000_000.0,
                recent_return: -0.04, volatility: 0.06,
                indicators: vec![("rsi".into(), 42.0), ("fear_greed".into(), 0.30)],
                news: vec![
                    news("REUTERS", "Peoples Bank China bans cryptocurrency transactions illegal", -0.90, 2.0),
                ],
                timestamp: ts(10, 4),
            },
            outcome: Outcome::new(Direction::Down, 0.07),
        },
        HistoricalEvent {
            label: "First BTC futures ETF approval",
            date: "2021-10-15",
            state: MarketState {
                price: 60000.0, volume: 65_000_000_000.0,
                recent_return: 0.10, volatility: 0.05,
                indicators: vec![("rsi".into(), 75.0), ("fear_greed".into(), 0.78)],
                news: vec![
                    news("SEC", "SEC approves first bitcoin futures ETF BITO ProShares", 0.85, 3.0),
                ],
                timestamp: ts(15, 4),
            },
            outcome: Outcome::new(Direction::Up, 0.08),
        },
        HistoricalEvent {
            label: "BTC all-time high $69k",
            date: "2021-11-10",
            state: MarketState {
                price: 69000.0, volume: 80_000_000_000.0,
                recent_return: 0.06, volatility: 0.04,
                indicators: vec![("rsi".into(), 84.0), ("fear_greed".into(), 0.90)],
                news: vec![
                    news("CPI", "US inflation print hottest decades 6.2 percent annual", 0.50, 8.0),
                ],
                timestamp: ts(14, 2),
            },
            outcome: Outcome::new(Direction::Down, 0.06),
        },
        // ============= 2022: Bear market =============
        HistoricalEvent {
            label: "Russia invades Ukraine",
            date: "2022-02-24",
            state: MarketState {
                price: 36000.0, volume: 55_000_000_000.0,
                recent_return: -0.08, volatility: 0.09,
                indicators: vec![("rsi".into(), 32.0), ("fear_greed".into(), 0.22)],
                news: vec![
                    news("REUTERS", "Russia full scale invasion Ukraine war begins", -0.85, 4.0),
                    news("MACRO", "Risk assets sell off oil gold spike safe haven", -0.50, 6.0),
                ],
                timestamp: ts(7, 3),
            },
            outcome: Outcome::new(Direction::Up, 0.10),
        },
        HistoricalEvent {
            label: "Fed first rate hike of cycle",
            date: "2022-03-16",
            state: MarketState {
                price: 41000.0, volume: 40_000_000_000.0,
                recent_return: 0.02, volatility: 0.05,
                indicators: vec![("rsi".into(), 60.0), ("fear_greed".into(), 0.35)],
                news: vec![
                    news("FED", "Fed raises federal funds rate 25 basis points first hike since 2018", 0.30, 1.0),
                    news("FED", "Powell hawkish statement six more hikes expected this year", -0.60, 0.5),
                ],
                timestamp: ts(18, 2),
            },
            outcome: Outcome::new(Direction::Down, 0.05),
        },
        HistoricalEvent {
            label: "LUNA / Terra collapse",
            date: "2022-05-09",
            state: MarketState {
                price: 32000.0, volume: 70_000_000_000.0,
                recent_return: -0.10, volatility: 0.10,
                indicators: vec![("rsi".into(), 28.0), ("fear_greed".into(), 0.18)],
                news: vec![
                    news("ONCHAIN", "Terra UST stablecoin depeg LUNA hyperinflation collapse", -0.95, 6.0),
                    news("CORP", "Three Arrows Capital insolvency contagion DeFi liquidations", -0.85, 12.0),
                ],
                timestamp: ts(14, 0),
            },
            outcome: Outcome::new(Direction::Down, 0.20),
        },
        HistoricalEvent {
            label: "Fed 75 bps hike — biggest since 1994",
            date: "2022-06-15",
            state: MarketState {
                price: 21000.0, volume: 55_000_000_000.0,
                recent_return: -0.15, volatility: 0.10,
                indicators: vec![("rsi".into(), 22.0), ("fear_greed".into(), 0.10)],
                news: vec![
                    news("FED", "Fed raises rates 75 basis points biggest hike since 1994 inflation fight", -0.65, 0.5),
                    news("CPI", "US CPI 8.6 percent 41 year high", -0.70, 36.0),
                ],
                timestamp: ts(18, 2),
            },
            outcome: Outcome::new(Direction::Up, 0.04),
        },
        HistoricalEvent {
            label: "Ethereum merge to PoS",
            date: "2022-09-15",
            state: MarketState {
                price: 20000.0, volume: 30_000_000_000.0,
                recent_return: -0.03, volatility: 0.05,
                indicators: vec![("rsi".into(), 45.0)],
                news: vec![
                    news("ONCHAIN", "Ethereum merge proof of stake successful transition", 0.50, 2.0),
                ],
                timestamp: ts(7, 3),
            },
            outcome: Outcome::new(Direction::Down, 0.03),
        },
        HistoricalEvent {
            label: "FTX collapse",
            date: "2022-11-08",
            state: MarketState {
                price: 19500.0, volume: 90_000_000_000.0,
                recent_return: -0.12, volatility: 0.13,
                indicators: vec![("rsi".into(), 18.0), ("fear_greed".into(), 0.12)],
                news: vec![
                    news("CORP", "FTX bankruptcy filing customer funds missing Sam Bankman Fried", -0.95, 4.0),
                    news("TWITTER", "Coindesk Alameda balance sheet leak FTT token solvency crisis", -0.85, 36.0),
                ],
                timestamp: ts(14, 1),
            },
            outcome: Outcome::new(Direction::Down, 0.18),
        },
        // ============= 2023: Recovery, banking crisis, BlackRock ETF =============
        HistoricalEvent {
            label: "Silicon Valley Bank failure (haven trade)",
            date: "2023-03-10",
            state: MarketState {
                price: 22000.0, volume: 40_000_000_000.0,
                recent_return: -0.04, volatility: 0.07,
                indicators: vec![("rsi".into(), 38.0), ("fear_greed".into(), 0.30)],
                news: vec![
                    news("REUTERS", "Silicon Valley Bank seized FDIC tech startups deposits frozen", -0.80, 6.0),
                    news("MACRO", "USDC stablecoin depeg Circle SVB exposure", -0.70, 12.0),
                ],
                timestamp: ts(15, 4),
            },
            outcome: Outcome::new(Direction::Up, 0.20),
        },
        HistoricalEvent {
            label: "BlackRock files for spot BTC ETF",
            date: "2023-06-15",
            state: MarketState {
                price: 26000.0, volume: 18_000_000_000.0,
                recent_return: 0.01, volatility: 0.03,
                indicators: vec![("rsi".into(), 55.0), ("fear_greed".into(), 0.50)],
                news: vec![
                    news("SEC", "BlackRock files spot bitcoin ETF iShares trust application", 0.90, 2.0),
                ],
                timestamp: ts(15, 3),
            },
            outcome: Outcome::new(Direction::Up, 0.20),
        },
        HistoricalEvent {
            label: "Israel-Hamas conflict begins",
            date: "2023-10-07",
            state: MarketState {
                price: 27500.0, volume: 14_000_000_000.0,
                recent_return: 0.02, volatility: 0.04,
                indicators: vec![("rsi".into(), 58.0)],
                news: vec![
                    news("REUTERS", "Hamas attack Israel southern conflict escalation Gaza", -0.65, 8.0),
                ],
                timestamp: ts(8, 5),
            },
            outcome: Outcome::new(Direction::Up, 0.04),
        },
        HistoricalEvent {
            label: "ETF approval rumor (false report)",
            date: "2023-10-16",
            state: MarketState {
                price: 28000.0, volume: 50_000_000_000.0,
                recent_return: 0.05, volatility: 0.08,
                indicators: vec![("rsi".into(), 70.0), ("fear_greed".into(), 0.65)],
                news: vec![
                    news("TWITTER", "Cointelegraph BlackRock spot ETF approved tweet retracted false", 0.50, 1.0),
                ],
                timestamp: ts(15, 0),
            },
            outcome: Outcome::new(Direction::Down, 0.03),
        },
        // ============= 2024: ETF approval, halving, ATH =============
        HistoricalEvent {
            label: "Spot BTC ETF approved",
            date: "2024-01-10",
            state: MarketState {
                price: 46000.0, volume: 35_000_000_000.0,
                recent_return: 0.04, volatility: 0.04,
                indicators: vec![("rsi".into(), 72.0), ("fear_greed".into(), 0.78)],
                news: vec![
                    news("SEC", "SEC approves eleven spot bitcoin ETFs Grayscale BlackRock Fidelity", 0.85, 4.0),
                    news("TWITTER", "Gensler statement spot bitcoin ETF approval reluctant order", 0.70, 3.0),
                ],
                timestamp: ts(22, 2),
            },
            outcome: Outcome::new(Direction::Down, 0.15),
        },
        HistoricalEvent {
            label: "ETF flows ATH approach",
            date: "2024-03-13",
            state: MarketState {
                price: 73000.0, volume: 70_000_000_000.0,
                recent_return: 0.10, volatility: 0.06,
                indicators: vec![("rsi".into(), 88.0), ("fear_greed".into(), 0.92)],
                news: vec![
                    news("CORP", "Spot bitcoin ETF inflows record day BlackRock IBIT volume", 0.80, 4.0),
                ],
                timestamp: ts(15, 2),
            },
            outcome: Outcome::new(Direction::Down, 0.10),
        },
        HistoricalEvent {
            label: "Bitcoin halving 2024",
            date: "2024-04-19",
            state: MarketState {
                price: 64000.0, volume: 30_000_000_000.0,
                recent_return: -0.03, volatility: 0.04,
                indicators: vec![("rsi".into(), 50.0)],
                news: vec![
                    news("ONCHAIN", "Bitcoin fourth halving block reward 3.125 BTC supply shock", 0.55, 0.5),
                ],
                timestamp: ts(0, 5),
            },
            outcome: Outcome::new(Direction::Flat, 0.01),
        },
        HistoricalEvent {
            label: "Mt Gox repayment fears",
            date: "2024-07-05",
            state: MarketState {
                price: 56000.0, volume: 45_000_000_000.0,
                recent_return: -0.07, volatility: 0.08,
                indicators: vec![("rsi".into(), 30.0), ("fear_greed".into(), 0.25)],
                news: vec![
                    news("ONCHAIN", "Mt Gox trustee bitcoin repayment creditors 140 thousand BTC", -0.70, 12.0),
                    news("CORP", "German government bitcoin sales seized funds liquidation pressure", -0.60, 24.0),
                ],
                timestamp: ts(10, 4),
            },
            outcome: Outcome::new(Direction::Down, 0.05),
        },
        HistoricalEvent {
            label: "Fed rate cut (first since 2020)",
            date: "2024-09-18",
            state: MarketState {
                price: 60000.0, volume: 28_000_000_000.0,
                recent_return: 0.01, volatility: 0.04,
                indicators: vec![("rsi".into(), 52.0), ("fear_greed".into(), 0.50)],
                news: vec![
                    news("FED", "Fed cuts rates 50 basis points first cut since pandemic dovish", 0.65, 1.0),
                ],
                timestamp: ts(18, 2),
            },
            outcome: Outcome::new(Direction::Up, 0.05),
        },
        HistoricalEvent {
            label: "Trump election win",
            date: "2024-11-06",
            state: MarketState {
                price: 70000.0, volume: 60_000_000_000.0,
                recent_return: 0.05, volatility: 0.05,
                indicators: vec![("rsi".into(), 76.0), ("fear_greed".into(), 0.80)],
                news: vec![
                    news("REUTERS", "Trump wins presidential election crypto friendly policy expected", 0.85, 6.0),
                    news("TWITTER", "Crypto industry celebrates Trump victory regulatory pivot", 0.80, 5.0),
                ],
                timestamp: ts(8, 2),
            },
            outcome: Outcome::new(Direction::Up, 0.10),
        },
        HistoricalEvent {
            label: "BTC crosses $100k",
            date: "2024-12-05",
            state: MarketState {
                price: 100000.0, volume: 80_000_000_000.0,
                recent_return: 0.08, volatility: 0.05,
                indicators: vec![("rsi".into(), 84.0), ("fear_greed".into(), 0.92)],
                news: vec![
                    news("REUTERS", "Bitcoin first time crosses 100000 milestone psychological level", 0.75, 1.0),
                ],
                timestamp: ts(3, 3),
            },
            outcome: Outcome::new(Direction::Down, 0.04),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_size() {
        let events = historical_btc_events();
        assert!(events.len() >= 25,
            "Dataset should have at least 25 events, got {}", events.len());
    }

    #[test]
    fn test_dataset_chronological() {
        let events = historical_btc_events();
        let dates: Vec<&str> = events.iter().map(|e| e.date).collect();
        let mut sorted = dates.clone();
        sorted.sort();
        assert_eq!(dates, sorted,
            "Events must be in chronological order for clean train/test splits");
    }

    #[test]
    fn test_dataset_diverse_directions() {
        let events = historical_btc_events();
        let ups = events.iter().filter(|e| e.outcome.direction == Direction::Up).count();
        let downs = events.iter().filter(|e| e.outcome.direction == Direction::Down).count();
        assert!(ups > 5 && downs > 5,
            "Dataset must contain meaningful counts of Up and Down outcomes (ups={}, downs={})",
            ups, downs);
    }

    #[test]
    fn test_dataset_has_news_context() {
        let events = historical_btc_events();
        let with_news = events.iter().filter(|e| !e.state.news.is_empty()).count();
        assert_eq!(with_news, events.len(),
            "Every event should have news context (the whole point of this dataset)");
    }
}
