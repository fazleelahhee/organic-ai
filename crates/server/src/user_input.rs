use axum::Extension;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Deserialize)]
pub struct UserMessage {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UserResponse {
    pub received: bool,
    pub message: String,
}

/// Pending request: user question waiting for an answer.
#[derive(Debug)]
pub struct PendingRequest {
    pub question: String,
    pub response_tx: tokio::sync::oneshot::Sender<String>,
}

/// Shared channel for passing questions to the simulation thread.
pub type RequestQueue = Arc<Mutex<Vec<PendingRequest>>>;


pub async fn send_message(
    Extension(queue): Extension<RequestQueue>,
    Json(msg): Json<UserMessage>,
) -> Json<UserResponse> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    {
        let mut q = queue.lock().await;
        q.push(PendingRequest {
            question: msg.text.clone(),
            response_tx: tx,
        });
    }

    // Wait for the simulation thread to process (with timeout)
    match tokio::time::timeout(std::time::Duration::from_secs(60), rx).await {
        Ok(Ok(response)) => Json(UserResponse {
            received: true,
            message: response,
        }),
        Ok(Err(_)) => Json(UserResponse {
            received: true,
            message: "Organism is thinking...".to_string(),
        }),
        Err(_) => Json(UserResponse {
            received: true,
            message: "Organism took too long to respond. Try again.".to_string(),
        }),
    }
}
