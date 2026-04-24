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

        // Teach the organism language basics — like a parent teaching a child.
        // Each call is ONE exposure. More exposures = stronger learning.
        // The organism needs ~5 exposures to confidently learn a word.
        for _ in 0..6 {
            organic_neuron::language::teach_basics(&mut world.language);
        }
        println!("Language taught — vocabulary: {} words", world.language.vocabulary_size());

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
                    let memory_key = req.question.to_lowercase().trim().to_string();

                    // LAYER 0: Neural arithmetic — raw math symbols
                    let arithmetic_answer = world.arithmetic.try_compute(&req.question);

                    let (response, source) = if let Some(answer) = arithmetic_answer {
                        (answer, "neural arithmetic (computed)")

                    } else if let Some(expression) = world.language.try_understand(&req.question) {
                        // LAYER 1: Language cortex understood the words → converted to math
                        // Now compute the resulting expression
                        if let Some(answer) = world.arithmetic.try_compute(&expression) {
                            (answer, "language + arithmetic (understood & computed)")
                        } else {
                            // Language understood but arithmetic couldn't compute
                            // Fall through to cortex/memory/claude
                            if let Some(answer) = world.cortex.try_answer(&req.question) {
                                (answer, "cortex (learned)")
                            } else {
                                let memory_key = req.question.to_lowercase().trim().to_string();
                                let remembered = world.tool_handler.memory.retrieve(
                                    string_to_address(&memory_key)
                                );
                                if !remembered.is_empty() {
                                    let text = signals_to_text(&remembered);
                                    if !text.is_empty() {
                                        world.cortex.learn(&req.question, &text);
                                        (text, "memory")
                                    } else {
                                        let answer = ask_claude_and_learn(&req.question, &memory_key, &mut world);
                                        (answer, "claude (fallback)")
                                    }
                                } else {
                                    let answer = ask_claude_and_learn(&req.question, &memory_key, &mut world);
                                    (answer, "claude (fallback)")
                                }
                            }
                        }

                    } else if let Some(answer) = world.cortex.try_answer(&req.question) {
                        (answer, "cortex (learned)")

                    } else {
                        // LAYER 2: Check organism's memory
                        let remembered = world.tool_handler.memory.retrieve(
                            string_to_address(&memory_key)
                        );

                        if !remembered.is_empty() {
                            let text = signals_to_text(&remembered);
                            if !text.is_empty() {
                                // Teach cortex from memory too (reinforcement)
                                world.cortex.learn(&req.question, &text);
                                (text, "memory")
                            } else {
                                // Memory corrupted — fall to Claude
                                let answer = ask_claude_and_learn(&req.question, &memory_key, &mut world);
                                (answer, "claude (re-learned)")
                            }
                        } else {
                            // LAYER 3: Ask Claude (last resort)
                            let answer = ask_claude_and_learn(&req.question, &memory_key, &mut world);
                            (answer, "claude (first time)")
                        }
                    };

                    // Teach language cortex from every interaction
                    world.language.learn_from_interaction(&req.question, &response);

                    println!("  → Answered from: {} (vocab: {}, cortex exp: {})",
                        source, world.language.vocabulary_size(), world.cortex.experience());

                    world.session_memory.record_interaction(
                        world.tick_count,
                        &req.question,
                        match source {
                            "cortex (learned)" => 2.0,
                            "memory" => 1.0,
                            _ => 0.5,
                        },
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

/// Ask Claude and teach the organism (both memory and cortex).
fn ask_claude_and_learn(
    question: &str,
    memory_key: &str,
    world: &mut organic_engine::simulation::World,
) -> String {
    let result = organic_tools::external::llm_query(question);
    let answer = if result.success { result.output.clone() } else { "I don't know yet.".to_string() };

    // Store in memory
    let signals = text_to_store_signals(&answer);
    world.tool_handler.memory.store(string_to_address(memory_key), signals);

    // Teach the cortex — this is how it learns concepts
    world.cortex.learn(question, &answer);
    println!("  → Learned: stored in memory + taught cortex");

    answer
}

/// Convert a string to a memory address (hash to f32 in 0-1 range).
/// Uses a better hash to avoid collisions between similar strings.
fn string_to_address(s: &str) -> f32 {
    let mut hash = 5381u64;
    for (i, b) in s.bytes().enumerate() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64).wrapping_add(i as u64 * 257);
    }
    (hash % 1000000) as f32 / 1000000.0
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
