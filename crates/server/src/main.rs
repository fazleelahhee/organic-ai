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

    let request_queue = user_input::RequestQueue::default();
    let sim_queue = request_queue.clone();

    std::thread::spawn(move || {
        let save_path = organic_engine::persistence::default_save_path();
        let _ = std::fs::create_dir_all("data");

        let mut world = match organic_engine::persistence::load_world(&save_path) {
            Ok(world) => {
                println!("Loaded saved world — tick {}, {} organisms, brain trained {} times",
                    world.tick_count, world.organisms.len(), world.brain.total_training);
                world
            }
            Err(_) => {
                println!("No save found, creating new world");
                World::new(config)
            }
        };

        world.session_memory.record_session_start();

        // Teach language basics through exposure
        for _ in 0..6 {
            organic_neuron::language::teach_basics(&mut world.language);
        }

        let stats = world.brain.stats();
        println!("Brain: {} neurons, {} synapses, trained {} times",
            stats.total_neurons, stats.total_synapses, stats.total_training);
        println!("Language: {} words learned", world.language.vocabulary_size());

        loop {
            std::thread::sleep(std::time::Duration::from_millis(16));
            for _ in 0..10 {
                world.tick();
            }

            // Process pending user questions
            if let Ok(mut queue) = sim_queue.try_lock() {
                let requests: Vec<user_input::PendingRequest> = queue.drain(..).collect();
                drop(queue);

                for req in requests {
                    println!("Processing: \"{}\"", req.question);

                    // === PIPELINE ===
                    // 1. ARITHMETIC (instant) — raw math + language math
                    // 2. MEMORY (instant) — organism recalls from past experience
                    // 3. CLAUDE (fallback) — organism learns from the answer
                    // Brain trains in background on every answer.

                    let memory_key = req.question.to_lowercase().trim().to_string();

                    // Step 1: Neural math — spike population dynamics (genuine neural computation)
                    let math_answer = world.neural_math.try_compute(&req.question)
                        .or_else(|| {
                            // Try language translation for word-based math
                            world.language.try_understand(&req.question)
                                .and_then(|expr| world.neural_math.try_compute(&expr))
                        });

                    let (response, source) = if let Some(answer) = math_answer {
                        (answer, "arithmetic")
                    } else {
                        // Step 2: Check memory — has the organism seen this before?
                        let remembered = world.tool_handler.memory.retrieve(
                            string_to_address(&memory_key)
                        );
                        let mem_text = if !remembered.is_empty() {
                            let t = signals_to_text(&remembered);
                            if !t.is_empty() { Some(t) } else { None }
                        } else { None };

                        if let Some(answer) = mem_text {
                            (answer, "memory (recalled)")
                        } else {
                            // Step 3: Ask Claude — organism learns
                            let result = organic_tools::external::llm_query(&req.question);
                            let answer = if result.success {
                                result.output.clone()
                            } else {
                                "I don't know yet.".to_string()
                            };
                            // Store in memory for next time
                            let signals = text_to_store_signals(&answer);
                            world.tool_handler.memory.store(
                                string_to_address(&memory_key), signals
                            );
                            (answer, "claude (learned)")
                        }
                    };

                    // Send response IMMEDIATELY — don't block on brain training
                    let _ = req.response_tx.send(response.clone());

                    println!("  → {} (response sent)", source);

                    // Learn AFTER responding (non-blocking for the user)
                    world.language.learn_from_interaction(&req.question, &response);
                    world.session_memory.record_interaction(
                        world.tick_count, &req.question,
                        if source.starts_with("arithmetic") { 1.5 } else { 0.5 },
                    );

                    // Train brain in background — this is slow at 40M but doesn't block response
                    world.brain.train(&req.question, &response);
                    println!("  → brain trained (total: {})", world.brain.total_training);
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

fn string_to_address(s: &str) -> f32 {
    let mut hash = 5381u64;
    for (i, b) in s.bytes().enumerate() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64).wrapping_add(i as u64 * 257);
    }
    (hash % 1000000) as f32 / 1000000.0
}

fn text_to_store_signals(text: &str) -> Vec<f32> {
    text.bytes().take(200).map(|b| b as f32 / 255.0).collect()
}

fn signals_to_text(signals: &[f32]) -> String {
    signals.iter()
        .map(|&s| (s * 255.0) as u8)
        .filter(|&b| b >= 32 && b < 127)
        .map(|b| b as char)
        .collect::<String>()
        .trim()
        .to_string()
}
