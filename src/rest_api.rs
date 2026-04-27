use axum::{Router, routing::{get, post}, Json, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use crate::audit::AuditChain;
use crate::vector_db::VectorDb;
use crate::fed_server::FedServer;

#[derive(Clone)]
pub struct AppState {
    pub audit: Arc<AuditChain>,
    pub vector_db: Arc<Mutex<VectorDb>>,
    pub fed_server: Arc<FedServer>,
}

#[derive(Serialize)]
struct HealthResponse { status: String, version: String, chain_length: i64 }

#[derive(Deserialize)]
struct FingerprintQuery { query_vector: Vec<f32>, top_k: Option<usize> }

#[derive(Serialize)]
struct PatternMatch { pattern_id: String, similarity: f32 }

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/v1/fingerprint/search", post(search_fingerprint))
        .route("/api/v1/audit/verify", get(verify_chain))
        .route("/api/v1/audit/chain", get(get_chain))
        .route("/api/v1/fed/round/start", post(start_round))
        .route("/api/v1/fed/round/aggregate", post(aggregate_round))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string(), version: "0.2.0".to_string(), chain_length: state.audit.chain_length().unwrap_or(0) })
}

async fn search_fingerprint(State(state): State<AppState>, Json(query): Json<FingerprintQuery>) -> Json<Vec<PatternMatch>> {
    let db = state.vector_db.lock().unwrap();
    let top_k = query.top_k.unwrap_or(5);
    let results = db.search(&query.query_vector, top_k);
    Json(results.into_iter().map(|(id, sim)| PatternMatch { pattern_id: id, similarity: sim }).collect())
}

async fn verify_chain(State(state): State<AppState>) -> Json<serde_json::Value> {
    match state.audit.verify_chain() {
        Ok((valid, length)) => Json(serde_json::json!({ "valid": valid, "chain_length": length })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn get_chain(State(state): State<AppState>) -> Json<serde_json::Value> {
    match state.audit.get_recent(20, None) {
        Ok(entries) => Json(serde_json::json!({ "entries": entries })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
struct RoundBody { round_num: Option<u64> }

async fn start_round(State(state): State<AppState>, Json(_body): Json<RoundBody>) -> Json<serde_json::Value> {
    match state.fed_server.start_round() {
        Ok(round_id) => Json(serde_json::json!({ "success": true, "round_id": round_id })),
        Err(e) => Json(serde_json::json!({ "error": e })),
    }
}

async fn aggregate_round(State(state): State<AppState>) -> Json<serde_json::Value> {
    let round_id = state.fed_server.current_round();
    match state.fed_server.aggregate(round_id) {
        Ok(result) => Json(serde_json::json!({ "success": true, "global_loss": result.global_loss, "clients": result.participating_clients })),
        Err(e) => Json(serde_json::json!({ "error": e })),
    }
}
