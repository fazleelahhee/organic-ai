/// Network protocol — messages exchanged between brain nodes.
///
/// Designed for the organic brain's distributed mode:
/// - Spike events cross node boundaries
/// - Weight updates sync learned knowledge
/// - Nodes discover each other and negotiate neuron ranges
///
/// The protocol is minimal: real brains don't send complex messages
/// between hemispheres — just spike trains. We do the same.

use serde::{Deserialize, Serialize};

/// Unique identifier for a node in the distributed brain.
pub type NodeId = u32;

/// Range of neuron indices owned by a node.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuronRange {
    pub start: u32,
    pub end: u32, // exclusive
}

impl NeuronRange {
    pub fn contains(&self, idx: u32) -> bool {
        idx >= self.start && idx < self.end
    }

    pub fn count(&self) -> u32 {
        self.end - self.start
    }
}

/// Messages between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Node announces itself and its neuron range.
    Hello {
        node_id: NodeId,
        range: NeuronRange,
        address: String, // "host:port"
    },

    /// Spike events from neurons that have cross-node connections.
    /// Sent every tick — only includes neurons that fired.
    Spikes {
        tick: u64,
        fired_indices: Vec<u32>, // neuron indices that fired this tick
    },

    /// Periodic weight sync — share learned synapse weights.
    /// Nodes merge incoming weights with their own (average).
    /// This is how distributed learning converges.
    WeightSync {
        tick: u64,
        /// (neuron_idx, synapse_idx, weight)
        weights: Vec<(u32, u16, f32)>,
    },

    /// Query — one node asks the brain to process a question.
    /// The hosting node coordinates the response.
    Query {
        id: u64,
        text: String,
    },

    /// Response to a query.
    QueryResponse {
        id: u64,
        text: String,
        source_node: NodeId,
    },

    /// Heartbeat — nodes ping each other to stay connected.
    Ping { node_id: NodeId, tick: u64 },
    Pong { node_id: NodeId, tick: u64 },

    /// Shutdown — node is leaving the cluster.
    Goodbye { node_id: NodeId },
}

/// Serialize a message to bytes for network transmission.
pub fn encode(msg: &Message) -> Vec<u8> {
    // Length-prefixed bincode
    let data = bincode::serialize(msg).unwrap_or_default();
    let len = data.len() as u32;
    let mut buf = len.to_le_bytes().to_vec();
    buf.extend_from_slice(&data);
    buf
}

/// Deserialize a message from bytes.
pub fn decode(buf: &[u8]) -> Option<Message> {
    if buf.len() < 4 { return None; }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < 4 + len { return None; }
    bincode::deserialize(&buf[4..4 + len]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let msg = Message::Spikes {
            tick: 42,
            fired_indices: vec![10, 20, 30],
        };
        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();
        match decoded {
            Message::Spikes { tick, fired_indices } => {
                assert_eq!(tick, 42);
                assert_eq!(fired_indices, vec![10, 20, 30]);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_neuron_range() {
        let range = NeuronRange { start: 100, end: 200 };
        assert!(range.contains(100));
        assert!(range.contains(199));
        assert!(!range.contains(200));
        assert_eq!(range.count(), 100);
    }

    #[test]
    fn test_encode_decode_weight_sync() {
        let msg = Message::WeightSync {
            tick: 100,
            weights: vec![(5, 2, 0.75), (10, 0, 0.33)],
        };
        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();
        match decoded {
            Message::WeightSync { tick, weights } => {
                assert_eq!(tick, 100);
                assert_eq!(weights.len(), 2);
            }
            _ => panic!("Wrong message type"),
        }
    }
}
