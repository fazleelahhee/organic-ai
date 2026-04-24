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
                    // 2. CLAUDE (fallback) — trains the brain with the answer
                    // Brain training happens in background on every answer.

                    // Try arithmetic first (instant)
                    let math_answer = world.language.try_understand(&req.question)
                        .and_then(|expr| world.arithmetic.try_compute(&expr))
                        .or_else(|| world.arithmetic.try_compute(&req.question));

                    let (response, source) = if let Some(answer) = math_answer {
                        (answer, "arithmetic (computed)")
                    } else {
                        // Ask Claude
                        let result = organic_tools::external::llm_query(&req.question);
                        let answer = if result.success {
                            result.output.clone()
                        } else {
                            "I don't know yet.".to_string()
                        };
                        (answer, "claude")
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
