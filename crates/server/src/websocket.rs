use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use std::sync::Arc;
use tokio::sync::watch;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    state: axum::extract::State<Arc<watch::Receiver<String>>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.0))
}

async fn handle_socket(mut socket: WebSocket, rx: Arc<watch::Receiver<String>>) {
    let mut rx = (*rx).clone();
    loop {
        if rx.changed().await.is_err() {
            break;
        }
        let json = rx.borrow().clone();
        if socket.send(Message::Text(json.into())).await.is_err() {
            break;
        }
    }
}
