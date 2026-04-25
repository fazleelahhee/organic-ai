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
        println!("Brain: {} neurons, {} synapses", stats.total_neurons, stats.total_synapses);

        loop {
            std::thread::sleep(std::time::Duration::from_millis(16));
            for _ in 0..10 { world.tick(); }

            if let Ok(mut queue) = sim_queue.try_lock() {
                let requests: Vec<user_input::PendingRequest> = queue.drain(..).collect();
                drop(queue);

                for req in requests {
                    println!("Processing: \"{}\"", req.question);

                    // Brain tries. Claude teaches. Nothing else.
                    let brain_output = world.brain.process(&req.question);
                    let brain_answered = !brain_output.is_empty()
                        && brain_output.chars().any(|c| c.is_alphanumeric())
                        && world.brain.is_trained();

                    let (response, source) = if brain_answered {
                        (brain_output, "brain")
                    } else {
                        let result = organic_tools::external::llm_query(&req.question);
                        let answer = if result.success { result.output.clone() } else { "I don't know yet.".to_string() };
                        (answer, "claude")
                    };

                    let _ = req.response_tx.send(response.clone());
                    println!("  → {} | len={} | {:?}", source, response.len(), &response[..response.len().min(50)]);

                    // Only train when Claude teaches — don't overwrite brain's own answers.
                    if source == "claude" {
                        world.brain.train(&req.question, &response);
                    }
                    world.session_memory.record_interaction(
                        world.tick_count, &req.question,
                        if source == "brain" { 2.0 } else { 0.5 },
                    );
                }
            }

            // Let the brain think for itself
            if let Some(thought) = world.brain.tick_inner_life(world.tick_count) {
                println!("  [thought] {} → {}", thought.seed, thought.insight);
            }

            let snap = world.snapshot();
            if snap.tick % 1000 == 0 {
                println!("tick {} | organisms: {} | resources: {} | thoughts: {}",
                    snap.tick, snap.organism_count, snap.resource_count,
                    world.brain.inner_life.total_thoughts);
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

    println!("Server running at http://localhost:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
