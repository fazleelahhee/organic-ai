mod snapshot;
mod user_input;
mod websocket;

use axum::{routing::get, Router};
use organic_engine::simulation::{World, WorldConfig};
use std::sync::Arc;
use tokio::sync::watch;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[tokio::main]
async fn main() {
    let config = WorldConfig::default();
    println!(
        "OrganicAI — starting {}x{} world with {} organisms",
        config.width, config.height, config.initial_organism_count
    );

    let (snapshot_tx, snapshot_rx) = watch::channel(String::from("{}"));
    let snapshot_rx = Arc::new(snapshot_rx);

    // Shared request queue — user questions flow from server to simulation thread
    let request_queue = user_input::RequestQueue::default();
    let sim_queue = request_queue.clone();

    std::thread::spawn(move || {
        // Try to load saved world
        let save_path = organic_engine::persistence::default_save_path();
        let _ = std::fs::create_dir_all("data");

        let mut world = match organic_engine::persistence::load_world(&save_path) {
            Ok(world) => {
                println!("Loaded saved world — tick {}, {} organisms", world.tick_count, world.organisms.len());
                world
            }
            Err(_) => {
                println!("No save found, creating new world");
                World::new(config)
            }
        };

        world.session_memory.record_session_start();

        loop {
            std::thread::sleep(std::time::Duration::from_millis(16));
            for _ in 0..10 {
                world.tick();
            }

            // Process any pending user questions
            if let Ok(mut queue) = sim_queue.try_lock() {
                let requests: Vec<user_input::PendingRequest> = queue.drain(..).collect();
                drop(queue); // release lock before doing slow work

                for req in requests {
                    println!("Processing question: {}", req.question);

                    // Step 1: Check organism's memory — has it seen this before?
                    let memory_key = req.question.to_lowercase().trim().to_string();
                    let remembered = world.tool_handler.memory.retrieve(
                        string_to_address(&memory_key)
                    );

                    let (response, source) = if !remembered.is_empty() {
                        // Organism remembers! Convert stored signals back to text
                        let text = signals_to_text(&remembered);
                        if !text.is_empty() {
                            (text, "memory")
                        } else {
                            // Memory corrupted, fall through to Claude
                            let result = organic_tools::external::llm_query(&req.question);
                            let answer = if result.success { result.output.clone() } else { "I don't know yet.".to_string() };
                            // Learn: store in memory for next time
                            let signals = text_to_store_signals(&answer);
                            world.tool_handler.memory.store(string_to_address(&memory_key), signals);
                            (answer, "claude (re-learned)")
                        }
                    } else {
                        // Not in memory — ask Claude (use it as a tool)
                        let result = organic_tools::external::llm_query(&req.question);
                        let answer = if result.success { result.output.clone() } else { "I don't know yet.".to_string() };

                        // LEARN: store the answer in organism's memory
                        let signals = text_to_store_signals(&answer);
                        world.tool_handler.memory.store(string_to_address(&memory_key), signals);
                        println!("  → Learned and stored in memory");

                        (answer, "claude (first time)")
                    };

                    println!("  → Answered from: {}", source);

                    // Record the interaction
                    world.session_memory.record_interaction(
                        world.tick_count,
                        &req.question,
                        if source == "memory" { 1.0 } else { 0.5 },
                    );

                    let _ = req.response_tx.send(response);
                }
            }

            let snap = world.snapshot();
            if snap.tick % 1000 == 0 {
                println!(
                    "tick {} | organisms: {} | resources: {}",
                    snap.tick, snap.organism_count, snap.resource_count
                );
            }
            if snap.tick % 5000 == 0 && snap.tick > 0 {
                let _ = organic_engine::persistence::save_world(&world, &save_path);
                println!("Auto-saved at tick {}", snap.tick);
            }
            let json = snapshot::to_json(&snap);
            let _ = snapshot_tx.send(json);
        }
    });

    let app = Router::new()
        .route("/ws", get(websocket::ws_handler))
        .route("/api/message", axum::routing::post(user_input::send_message))
        .nest_service("/", ServeDir::new("web"))
        .layer(CorsLayer::permissive())
        .layer(axum::Extension(request_queue))
        .with_state(snapshot_rx);

    let addr = "0.0.0.0:3000";
    println!("Server running at http://localhost:3000");
    println!("WebSocket at ws://localhost:3000/ws");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Convert a string to a memory address (hash to f32 in 0-1 range).
fn string_to_address(s: &str) -> f32 {
    let hash = s.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    (hash % 10000) as f32 / 10000.0
}

/// Convert text to signal values for storage in organism memory.
fn text_to_store_signals(text: &str) -> Vec<f32> {
    text.bytes()
        .take(200) // limit stored length
        .map(|b| b as f32 / 255.0)
        .collect()
}

/// Convert stored signal values back to text.
fn signals_to_text(signals: &[f32]) -> String {
    signals.iter()
        .map(|&s| (s * 255.0) as u8)
        .filter(|&b| b >= 32 && b < 127) // printable ASCII only
        .map(|b| b as char)
        .collect::<String>()
        .trim()
        .to_string()
}
