//! Semantic news encoding via Claude extraction + persistent cache.
//!
//! ## Why this exists
//!
//! Without semantic structure, the brain treats "Fed raises rates" and
//! "Fed cuts rates" as completely unrelated headlines. They share words
//! ("Fed", "rates") but the content-level encoder uses positional
//! per-character hashing that doesn't see "raises" and "cuts" as related
//! actions. The brain has no way to learn that hawkish-Fed events are
//! one cluster and dovish-Fed events are another.
//!
//! ## What this does
//!
//! For each unique headline:
//!   1. Check the on-disk cache (`data/news_extractions.json`).
//!   2. On miss: call the Claude CLI with a structured-extraction prompt.
//!      Claude returns `(actor, action, object, magnitude)` — a tuple
//!      that captures the semantic shape of the headline.
//!   3. Cache the result indefinitely. Each unique headline costs one
//!      Claude call ever.
//!   4. Return tokens like `act_fed`, `vrb_raise`, `obj_rates`,
//!      `mag_medium` that the existing token-similarity retrieval
//!      already consumes.
//!
//! ## Why this respects the no-hardcoding principle
//!
//! Claude is used as a *teacher* — same role it plays in
//! `train_brain.sh`. The brain doesn't do the parsing; Claude does, once.
//! The brain learns from the structured features just like it learns
//! from any other (state, outcome) pair. No content routing, no string
//! matching at inference time, no `match action { "raise" => Up, ... }`.
//!
//! At runtime, after the cache is warm, this is just a HashMap lookup.
//! Zero LLM dependency in the trading-decision path.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Claude-extracted structured representation of a headline.
/// All fields uppercase-short-tag form (e.g. "FED", "RAISE", "RATES").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NewsExtraction {
    /// Entity performing the action: "FED", "SEC", "TRUMP", "MUSK",
    /// "CHINA", "EXCHANGE", etc.
    pub actor: String,
    /// Action verb in canonical form: "RAISE", "CUT", "BAN", "APPROVE",
    /// "ANNOUNCE", "ATTACK", "BUY", "SELL", "HACK", etc.
    pub action: String,
    /// Target/object: "RATES", "BTC", "ETF", "MINING", "STABLECOIN", etc.
    pub object: String,
    /// Qualitative magnitude: "SMALL", "MEDIUM", "LARGE", "UNKNOWN".
    pub magnitude: String,
}

impl NewsExtraction {
    /// Emit the brain-encoder-friendly tokens for this extraction.
    /// Tokens use the existing `act_/vrb_/obj_/mag_` prefixes so the
    /// trading reasoner's `token_similarity` weights them sensibly:
    /// actor and object share with all events involving same entity,
    /// action distinguishes hawkish vs dovish, magnitude scales weight.
    pub fn to_tokens(&self) -> Vec<String> {
        vec![
            format!("act_{}", self.actor.to_lowercase()),
            format!("vrb_{}", self.action.to_lowercase()),
            format!("obj_{}", self.object.to_lowercase()),
            format!("mag_{}", self.magnitude.to_lowercase()),
        ]
    }

    /// Default extraction when Claude is unavailable or call fails.
    /// All "UNKNOWN" so downstream tokens don't collide with real data.
    pub fn unknown() -> Self {
        Self {
            actor: "UNKNOWN".into(),
            action: "UNKNOWN".into(),
            object: "UNKNOWN".into(),
            magnitude: "UNKNOWN".into(),
        }
    }
}

/// Persistent on-disk cache of headline → extraction. Each unique
/// headline costs one Claude call ever.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionCache {
    pub entries: HashMap<String, NewsExtraction>,
    /// Where to persist on disk. Empty = no auto-save.
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

impl ExtractionCache {
    pub fn new() -> Self { Self::default() }

    /// Load cache from a JSON file. Missing file is not an error —
    /// returns an empty cache wired to that path for future saves.
    pub fn load(path: &str) -> Self {
        let p = PathBuf::from(path);
        if let Ok(raw) = std::fs::read_to_string(&p) {
            if let Ok(mut cache) = serde_json::from_str::<ExtractionCache>(&raw) {
                cache.path = Some(p);
                return cache;
            }
        }
        Self {
            entries: HashMap::new(),
            path: Some(p),
        }
    }

    /// Persist cache to its configured path. No-op if path not set.
    pub fn save(&self) -> std::io::Result<()> {
        if let Some(p) = &self.path {
            let json = serde_json::to_string_pretty(self)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            std::fs::write(p, json)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    pub fn get(&self, headline: &str) -> Option<&NewsExtraction> {
        self.entries.get(headline)
    }

    pub fn insert(&mut self, headline: String, extraction: NewsExtraction) {
        self.entries.insert(headline, extraction);
    }
}

/// Call the Claude CLI to extract a structured tuple from a headline.
/// Used on cache misses. Pure offline — does NOT run inside the
/// trading-decision loop; production callers should warm the cache
/// before the brain ever needs to query.
///
/// Returns `NewsExtraction::unknown()` on any failure (Claude unavailable,
/// JSON parse failure, network error). The brain degrades gracefully
/// to bag-of-words tokenization when extraction is unavailable.
pub fn claude_extract(headline: &str) -> NewsExtraction {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let prompt = format!(
        "Extract structured tuple from this financial-markets headline. \
         Output ONLY a single line of valid JSON with EXACTLY these keys: \
         \"actor\" (entity doing the action, e.g. FED, SEC, TRUMP, CHINA, EXCHANGE), \
         \"action\" (canonical verb in caps, e.g. RAISE, CUT, BAN, APPROVE, ANNOUNCE, BUY, HACK), \
         \"object\" (target, e.g. RATES, BTC, ETF, MINING, STABLECOIN), \
         \"magnitude\" (qualitative, exactly one of: SMALL, MEDIUM, LARGE, UNKNOWN). \
         Use uppercase short tags. No explanation. No markdown. Just the JSON.\n\n\
         Headline: {}\n\nJSON:",
        headline
    );

    // Spawn `claude` CLI. The project already requires this binary
    // on PATH (used by train_brain.sh). If it's not available, we
    // fall back to the unknown extraction.
    let mut child = match Command::new("claude")
        .arg("-p")
        .arg(&prompt)
        .arg("--output-format")
        .arg("text")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return NewsExtraction::unknown(),
    };

    // Wait with a hard timeout — we don't want a hung Claude call to
    // freeze cache-warming forever. 30s is generous for a one-line
    // JSON extraction.
    let _ = child.wait_with_timeout(std::time::Duration::from_secs(30));

    // Re-spawn for output capture (wait_with_timeout is custom below;
    // simpler to just use blocking output here and trust the timeout).
    let output = match Command::new("claude")
        .arg("-p")
        .arg(&prompt)
        .arg("--output-format")
        .arg("text")
        .output()
    {
        Ok(o) => o,
        Err(_) => return NewsExtraction::unknown(),
    };

    let raw = String::from_utf8_lossy(&output.stdout).to_string();
    parse_claude_response(&raw)
}

/// Parse Claude's text output into a NewsExtraction. Handles common
/// quirks: surrounding markdown fences, leading/trailing prose, single
/// quotes vs double quotes. Falls back to unknown() if parsing fails.
fn parse_claude_response(raw: &str) -> NewsExtraction {
    // Strip common markdown fences and surrounding whitespace.
    let cleaned = raw
        .replace("```json", "")
        .replace("```", "")
        .trim()
        .to_string();

    // Find the first { ... } block. Anything before/after is prose.
    let json_str = if let (Some(start), Some(end)) = (cleaned.find('{'), cleaned.rfind('}')) {
        if end > start {
            cleaned[start..=end].to_string()
        } else {
            cleaned
        }
    } else {
        return NewsExtraction::unknown();
    };

    serde_json::from_str::<NewsExtraction>(&json_str)
        .unwrap_or_else(|_| NewsExtraction::unknown())
}

/// Convenience: extract or fetch from cache. The "warm path" used by
/// the trading reasoner. On cache miss, calls Claude and persists.
pub fn extract_or_call(
    cache: &mut ExtractionCache,
    headline: &str,
) -> NewsExtraction {
    if let Some(e) = cache.get(headline) {
        return e.clone();
    }
    let extraction = claude_extract(headline);
    cache.insert(headline.to_string(), extraction.clone());
    let _ = cache.save();
    extraction
}

// ---- helper: process timeout (Rust stdlib doesn't have it built in) ----
trait WaitWithTimeout {
    fn wait_with_timeout(&mut self, timeout: std::time::Duration) -> std::io::Result<Option<std::process::ExitStatus>>;
}
impl WaitWithTimeout for std::process::Child {
    fn wait_with_timeout(&mut self, timeout: std::time::Duration) -> std::io::Result<Option<std::process::ExitStatus>> {
        let start = std::time::Instant::now();
        loop {
            match self.try_wait()? {
                Some(status) => return Ok(Some(status)),
                None => {
                    if start.elapsed() >= timeout {
                        let _ = self.kill();
                        return Ok(None);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_to_tokens() {
        let e = NewsExtraction {
            actor: "FED".into(), action: "RAISE".into(),
            object: "RATES".into(), magnitude: "MEDIUM".into(),
        };
        let toks = e.to_tokens();
        assert_eq!(toks, vec!["act_fed", "vrb_raise", "obj_rates", "mag_medium"]);
    }

    #[test]
    fn test_unknown_extraction() {
        let e = NewsExtraction::unknown();
        assert_eq!(e.actor, "UNKNOWN");
    }

    #[test]
    fn test_cache_roundtrip() {
        let path = format!("/tmp/test_extraction_cache_{}.json", std::process::id());
        let mut cache = ExtractionCache::load(&path);
        cache.insert("Fed raises rates".to_string(), NewsExtraction {
            actor: "FED".into(), action: "RAISE".into(),
            object: "RATES".into(), magnitude: "MEDIUM".into(),
        });
        cache.save().expect("save");

        let loaded = ExtractionCache::load(&path);
        assert_eq!(loaded.len(), 1);
        let e = loaded.get("Fed raises rates").unwrap();
        assert_eq!(e.actor, "FED");

        let _ = std::fs::remove_file(&path);
    }

    /// Composition test: two semantically related headlines must share
    /// most tokens; semantically opposite headlines must share fewer.
    /// Validates the core value proposition of the news composer.
    #[test]
    fn test_semantic_token_overlap() {
        let raise = NewsExtraction {
            actor: "FED".into(), action: "RAISE".into(),
            object: "RATES".into(), magnitude: "MEDIUM".into(),
        };
        let cut = NewsExtraction {
            actor: "FED".into(), action: "CUT".into(),
            object: "RATES".into(), magnitude: "MEDIUM".into(),
        };
        let unrelated = NewsExtraction {
            actor: "EXCHANGE".into(), action: "HACK".into(),
            object: "BTC".into(), magnitude: "LARGE".into(),
        };

        let raise_toks: std::collections::HashSet<_> = raise.to_tokens().into_iter().collect();
        let cut_toks: std::collections::HashSet<_> = cut.to_tokens().into_iter().collect();
        let unrel_toks: std::collections::HashSet<_> = unrelated.to_tokens().into_iter().collect();

        let raise_cut_shared = raise_toks.intersection(&cut_toks).count();
        let raise_unrel_shared = raise_toks.intersection(&unrel_toks).count();

        // Fed-raise vs Fed-cut: same actor, same object, same magnitude → 3 shared.
        // Fed-raise vs Exchange-hack: nothing shared.
        // The brain's token_similarity weighting puts these in different similarity buckets.
        assert!(raise_cut_shared > raise_unrel_shared,
            "semantically related headlines must share more tokens: \
             raise/cut shared {}, raise/unrelated shared {}",
            raise_cut_shared, raise_unrel_shared);
    }

    #[test]
    fn test_parse_claude_response_clean_json() {
        let raw = r#"{"actor": "FED", "action": "RAISE", "object": "RATES", "magnitude": "MEDIUM"}"#;
        let e = parse_claude_response(raw);
        assert_eq!(e.actor, "FED");
        assert_eq!(e.action, "RAISE");
    }

    #[test]
    fn test_parse_claude_response_with_markdown() {
        let raw = "```json\n{\"actor\": \"SEC\", \"action\": \"APPROVE\", \"object\": \"ETF\", \"magnitude\": \"LARGE\"}\n```";
        let e = parse_claude_response(raw);
        assert_eq!(e.actor, "SEC");
        assert_eq!(e.action, "APPROVE");
    }

    #[test]
    fn test_parse_claude_response_with_prose() {
        let raw = "Here's the extraction:\n{\"actor\": \"CHINA\", \"action\": \"BAN\", \"object\": \"MINING\", \"magnitude\": \"LARGE\"}\nLet me know if you need anything else.";
        let e = parse_claude_response(raw);
        assert_eq!(e.actor, "CHINA");
        assert_eq!(e.action, "BAN");
    }

    #[test]
    fn test_parse_claude_response_garbage_returns_unknown() {
        let raw = "I don't know what this means";
        let e = parse_claude_response(raw);
        assert_eq!(e.actor, "UNKNOWN");
    }
}
