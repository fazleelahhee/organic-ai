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

                    // === THE GENUINE PIPELINE ===
                    //
                    // 1. BRAIN: Process through 2048-neuron spiking network (always)
                    // 2. LANGUAGE + ARITHMETIC: If brain output is weak, try
                    //    language translation → spike-based arithmetic
                    // 3. CLAUDE: Last resort — but we use the answer to TRAIN
                    //    the brain, not just cache it

                    // Step 1: Let the brain try
                    let brain_output = world.brain.process(&req.question);
                    let brain_confident = !brain_output.is_empty()
                        && brain_output.len() > 0
                        && world.brain.is_trained();

                    let (response, source) = if brain_confident && brain_output.chars().any(|c| c.is_alphanumeric()) {
                        // Brain produced a meaningful response
                        (brain_output, "brain (spiking network)")
                    } else {
                        // Step 2: Try language → arithmetic path for math
                        let math_answer = if let Some(expr) = world.language.try_understand(&req.question) {
                            world.arithmetic.try_compute(&expr)
                        } else {
                            world.arithmetic.try_compute(&req.question)
                        };

                        if let Some(answer) = math_answer {
                            // Train the brain with this correct answer
                            world.brain.train(&req.question, &answer);
                            (answer, "arithmetic (computed) → trained brain")
                        } else {
                            // Step 3: Ask Claude — use answer to train the brain
                            let result = organic_tools::external::llm_query(&req.question);
                            let answer = if result.success {
                                result.output.clone()
                            } else {
                                "I don't know yet.".to_string()
                            };

                            // TRAIN the brain with Claude's answer
                            // This is the key: Claude is a teacher, not a crutch.
                            // The brain's STDP will wire the association between
                            // the question's spike pattern and the answer's spike pattern.
                            for _ in 0..3 { // multiple exposures to strengthen
                                world.brain.train(&req.question, &answer);
                            }

                            (answer, "claude → trained brain (3x)")
                        }
                    };

                    // Teach language from interaction
                    world.language.learn_from_interaction(&req.question, &response);

                    let stats = world.brain.stats();
                    println!("  → {} | brain: {} trained, {} active neurons, avg_w: {:.4}",
                        source, stats.total_training, stats.active_neurons, stats.avg_weight);

                    world.session_memory.record_interaction(
                        world.tick_count,
                        &req.question,
                        if source.starts_with("brain") { 2.0 }
                        else if source.starts_with("arithmetic") { 1.5 }
                        else { 0.5 },
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
