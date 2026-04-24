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

        let stats = world.brain.stats();
        println!("Brain: {} neurons, {} synapses, trained {} times",
            stats.total_neurons, stats.total_synapses, stats.total_training);

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

                    // === ORGANIC PIPELINE ===
                    // ONE path: the brain processes everything.
                    // If the brain can't answer → Claude teaches it.
                    // No routing. No if/else. No parsers. No dictionaries.
                    // The brain IS the intelligence.

                    // Let the brain try to answer
                    let brain_output = world.brain.process(&req.question);
                    let brain_has_answer = !brain_output.is_empty()
                        && brain_output.chars().any(|c| c.is_alphanumeric())
                        && world.brain.is_trained();

                    let (response, source) = if brain_has_answer {
                        // Brain answered from its own spiking network
                        (brain_output, "brain")
                    } else {
                        // Brain doesn't know yet — ask Claude to TEACH it
                        let result = organic_tools::external::llm_query(&req.question);
                        let answer = if result.success {
                            result.output.clone()
                        } else {
                            "I don't know yet.".to_string()
                        };
                        (answer, "claude (teaching)")
                    };

                    // Send response immediately
                    let _ = req.response_tx.send(response.clone());
                    println!("  → {} (response sent)", source);

                    // ALWAYS train the brain — whether it answered or Claude did.
                    // When the brain answered: reinforcement (strengthen existing pathways)
                    // When Claude answered: new learning (form new pathways)
                    world.brain.train(&req.question, &response);

                    world.session_memory.record_interaction(
                        world.tick_count, &req.question,
                        if source == "brain" { 2.0 } else { 0.5 },
                    );

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
