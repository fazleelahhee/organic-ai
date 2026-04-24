use organic_engine::simulation::WorldSnapshot;

pub fn to_json(snapshot: &WorldSnapshot) -> String {
    serde_json::to_string(snapshot).unwrap_or_else(|_| "{}".to_string())
}
