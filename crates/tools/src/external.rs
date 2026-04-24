/// Result of an external tool query.
#[derive(Debug, Clone)]
pub struct ExternalResult {
    pub output: String,
    pub success: bool,
    pub signal_value: f32,
}

/// Web search via a simple command-line approach.
/// In production this would use an API, but for M5 we use a simple local approach.
pub fn web_search(query: &str) -> ExternalResult {
    // Use curl to search (simplified — returns a signal indicating "found something")
    // For safety, we don't actually execute arbitrary searches.
    // Instead, we simulate the interface and return a placeholder.
    ExternalResult {
        output: format!("Search results for: {}", query),
        success: true,
        signal_value: 0.7, // simulated relevance
    }
}

/// Query an LLM (simulated for M5 — real API integration in production).
pub fn llm_query(prompt: &str) -> ExternalResult {
    ExternalResult {
        output: format!("LLM response to: {}", prompt),
        success: true,
        signal_value: 0.8,
    }
}

/// Read a file from the filesystem (sandboxed to a safe directory).
pub fn read_file(filename: &str) -> ExternalResult {
    let safe_dir = "data/organism_files";
    let path = format!("{}/{}", safe_dir, filename.replace("..", "").replace("/", ""));

    match std::fs::read_to_string(&path) {
        Ok(contents) => ExternalResult {
            output: contents,
            success: true,
            signal_value: 1.0,
        },
        Err(_) => ExternalResult {
            output: String::new(),
            success: false,
            signal_value: 0.0,
        },
    }
}

/// Convert a signal pattern to a text query string.
pub fn signals_to_query(signals: &[f32]) -> String {
    signals.iter()
        .map(|s| {
            let idx = (s * 25.0) as u8 + b'a';
            (idx.min(b'z')) as char
        })
        .collect()
}

/// Convert text output back to signal values.
pub fn text_to_signals(text: &str, max_len: usize) -> Vec<f32> {
    text.bytes()
        .take(max_len)
        .map(|b| (b as f32 - b'a' as f32) / 25.0)
        .map(|v| v.clamp(0.0, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_returns_result() {
        let result = web_search("test query");
        assert!(result.success);
        assert!(result.signal_value > 0.0);
    }

    #[test]
    fn test_llm_query_returns_result() {
        let result = llm_query("what is 2+2?");
        assert!(result.success);
    }

    #[test]
    fn test_signals_to_query() {
        let signals = vec![0.0, 0.5, 1.0];
        let query = signals_to_query(&signals);
        assert_eq!(query.len(), 3);
        assert!(query.chars().all(|c| c.is_ascii_lowercase()));
    }

    #[test]
    fn test_text_to_signals() {
        let signals = text_to_signals("abc", 10);
        assert_eq!(signals.len(), 3);
        assert!(signals.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_roundtrip_signal_conversion() {
        let original = vec![0.0, 0.5, 1.0];
        let text = signals_to_query(&original);
        let recovered = text_to_signals(&text, 10);
        assert_eq!(original.len(), recovered.len());
    }

    #[test]
    fn test_file_read_nonexistent() {
        let result = read_file("nonexistent.txt");
        assert!(!result.success);
        assert_eq!(result.signal_value, 0.0);
    }
}
