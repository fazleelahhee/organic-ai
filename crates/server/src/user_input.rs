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

/// Shared state for user messages.
pub type MessageQueue = Arc<Mutex<Vec<String>>>;

pub async fn send_message(
    Extension(queue): Extension<MessageQueue>,
    Json(msg): Json<UserMessage>,
) -> Json<UserResponse> {
    let mut q = queue.lock().await;
    q.push(msg.text.clone());
    Json(UserResponse {
        received: true,
        message: format!("Message queued: {}", msg.text),
    })
}
