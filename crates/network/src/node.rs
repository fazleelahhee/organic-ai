/// Node — a single machine in the distributed brain.
///
/// Each node owns a range of neurons, processes them locally,
/// and communicates cross-boundary spikes with other nodes.
///
/// In standalone mode: one node owns all neurons.
/// In distributed mode: multiple nodes split the brain.

use crate::protocol::{self, Message, NeuronRange, NodeId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};

/// Connection to a peer node.
struct PeerConnection {
    node_id: NodeId,
    range: NeuronRange,
    sender: mpsc::Sender<Vec<u8>>,
}

/// A node in the distributed brain cluster.
pub struct BrainNode {
    pub node_id: NodeId,
    pub range: NeuronRange,
    pub address: String,
    peers: Arc<Mutex<HashMap<NodeId, PeerConnection>>>,
    /// Incoming spikes from other nodes (neuron_idx → fired)
    incoming_spikes: Arc<Mutex<Vec<u32>>>,
    /// Incoming weight updates from other nodes
    incoming_weights: Arc<Mutex<Vec<(u32, u16, f32)>>>,
}

impl BrainNode {
    /// Create a new node. In standalone mode, range covers all neurons.
    pub fn new(node_id: NodeId, range: NeuronRange, address: String) -> Self {
        Self {
            node_id,
            range,
            address,
            peers: Arc::new(Mutex::new(HashMap::new())),
            incoming_spikes: Arc::new(Mutex::new(Vec::new())),
            incoming_weights: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start listening for peer connections.
    pub async fn start_listener(&self) -> Result<(), String> {
        let addr = self.address.clone();
        let peers = self.peers.clone();
        let incoming_spikes = self.incoming_spikes.clone();
        let incoming_weights = self.incoming_weights.clone();

        tokio::spawn(async move {
            let listener = TcpListener::bind(&addr).await
                .map_err(|e| format!("Failed to bind: {}", e)).unwrap();

            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let spikes = incoming_spikes.clone();
                    let weights = incoming_weights.clone();
                    tokio::spawn(async move {
                        handle_peer(stream, spikes, weights).await;
                    });
                }
            }
        });

        Ok(())
    }

    /// Connect to a peer node.
    pub async fn connect_to_peer(&self, peer_addr: &str, peer_id: NodeId, peer_range: NeuronRange) -> Result<(), String> {
        let stream = TcpStream::connect(peer_addr).await
            .map_err(|e| format!("Failed to connect to {}: {}", peer_addr, e))?;

        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(1000);

        // Spawn writer task
        let (_, mut writer) = tokio::io::split(stream);
        tokio::spawn(async move {
            while let Some(data) = rx.recv().await {
                if writer.write_all(&data).await.is_err() { break; }
            }
        });

        // Send hello
        let hello = protocol::encode(&Message::Hello {
            node_id: self.node_id,
            range: self.range,
            address: self.address.clone(),
        });
        let _ = tx.send(hello).await;

        let mut peers = self.peers.lock().await;
        peers.insert(peer_id, PeerConnection {
            node_id: peer_id,
            range: peer_range,
            sender: tx,
        });

        Ok(())
    }

    /// Broadcast spikes to all peers.
    /// Only sends spikes for neurons that have cross-node connections.
    pub async fn broadcast_spikes(&self, tick: u64, fired_indices: Vec<u32>) {
        if fired_indices.is_empty() { return; }

        let msg = protocol::encode(&Message::Spikes { tick, fired_indices });
        let peers = self.peers.lock().await;
        for (_, peer) in peers.iter() {
            let _ = peer.sender.send(msg.clone()).await;
        }
    }

    /// Broadcast weight updates for distributed learning.
    pub async fn broadcast_weights(&self, tick: u64, weights: Vec<(u32, u16, f32)>) {
        if weights.is_empty() { return; }

        let msg = protocol::encode(&Message::WeightSync { tick, weights });
        let peers = self.peers.lock().await;
        for (_, peer) in peers.iter() {
            let _ = peer.sender.send(msg.clone()).await;
        }
    }

    /// Drain incoming spikes from peers.
    pub async fn drain_incoming_spikes(&self) -> Vec<u32> {
        let mut spikes = self.incoming_spikes.lock().await;
        std::mem::take(&mut *spikes)
    }

    /// Drain incoming weight updates from peers.
    pub async fn drain_incoming_weights(&self) -> Vec<(u32, u16, f32)> {
        let mut weights = self.incoming_weights.lock().await;
        std::mem::take(&mut *weights)
    }

    /// Check if running in distributed mode (has peers).
    pub async fn is_distributed(&self) -> bool {
        !self.peers.lock().await.is_empty()
    }

    pub async fn peer_count(&self) -> usize {
        self.peers.lock().await.len()
    }
}

/// Handle messages from a connected peer.
async fn handle_peer(
    stream: TcpStream,
    incoming_spikes: Arc<Mutex<Vec<u32>>>,
    incoming_weights: Arc<Mutex<Vec<(u32, u16, f32)>>>,
) {
    let (mut reader, _) = tokio::io::split(stream);
    let mut buf = vec![0u8; 65536];

    loop {
        match reader.read(&mut buf).await {
            Ok(0) => break, // connection closed
            Ok(n) => {
                if let Some(msg) = protocol::decode(&buf[..n]) {
                    match msg {
                        Message::Spikes { fired_indices, .. } => {
                            let mut spikes = incoming_spikes.lock().await;
                            spikes.extend_from_slice(&fired_indices);
                        }
                        Message::WeightSync { weights, .. } => {
                            let mut w = incoming_weights.lock().await;
                            w.extend_from_slice(&weights);
                        }
                        _ => {} // ignore other messages for now
                    }
                }
            }
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = BrainNode::new(0, NeuronRange { start: 0, end: 1000 }, "127.0.0.1:9000".to_string());
        assert_eq!(node.node_id, 0);
        assert_eq!(node.range.count(), 1000);
    }

    #[tokio::test]
    async fn test_standalone_mode() {
        let node = BrainNode::new(0, NeuronRange { start: 0, end: 2048 }, "127.0.0.1:9001".to_string());
        assert!(!node.is_distributed().await);
        assert_eq!(node.peer_count().await, 0);
    }

    #[tokio::test]
    async fn test_drain_empty_spikes() {
        let node = BrainNode::new(0, NeuronRange { start: 0, end: 100 }, "127.0.0.1:9002".to_string());
        let spikes = node.drain_incoming_spikes().await;
        assert!(spikes.is_empty());
    }
}
