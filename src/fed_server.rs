use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::audit::AuditChain;
use crate::vector_db::VectorDb;

/// Federated learning server for multi-workshop embroidery stitch optimization.
/// Mirrors embodied-fl/fed_server.rs — FedAvg aggregation with audit trail.
pub struct FedServer {
    global_model: HashMap<String, Vec<f32>>,  // stitch_type -> weights
    client_updates: Mutex<Vec<ClientUpdate>>,
    round_number: Mutex<u64>,
    audit: Arc<AuditChain>,
    vector_db: Arc<Mutex<VectorDb>>,
}

#[derive(Debug, Clone)]
pub struct ClientUpdate {
    pub client_id: String,
    pub round_id: u64,
    pub num_samples: u32,
    pub local_loss: f32,
    pub weights: Vec<f32>,
    pub stitch_type: String,
}

#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub global_loss: f32,
    pub participating_clients: usize,
}

impl FedServer {
    pub fn new(audit: Arc<AuditChain>, vector_db: Arc<Mutex<VectorDb>>) -> Self {
        Self {
            global_model: HashMap::new(),
            client_updates: Mutex::new(Vec::new()),
            round_number: Mutex::new(0),
            audit,
            vector_db,
        }
    }

    pub fn register_client(&self, client_id: &str, workshop_name: &str, specialty: &str) -> Result<(), String> {
        self.audit.add_entry("register_client", client_id, &format!("workshop={},specialty={}", workshop_name, specialty))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn start_round(&self) -> Result<u64, String> {
        let mut round = self.round_number.lock().unwrap();
        *round += 1;
        let round_id = *round;
        self.client_updates.lock().unwrap().clear();
        self.audit.add_entry("start_round", "server", &format!("round={}", round_id))
            .map_err(|e| e.to_string())?;
        Ok(round_id)
    }

    pub fn submit_update(&self, update: ClientUpdate) -> Result<(), String> {
        self.audit.add_entry("submit_update", &update.client_id,
            &format!("round={},samples={},loss={:.4},stitch={}", update.round_id, update.num_samples, update.local_loss, update.stitch_type))
            .map_err(|e| e.to_string())?;
        self.client_updates.lock().unwrap().push(update);
        Ok(())
    }

    pub fn aggregate(&self, round_id: u64) -> Result<AggregationResult, String> {
        let updates = self.client_updates.lock().unwrap();
        if updates.is_empty() {
            return Err("No updates to aggregate".to_string());
        }

        let total_samples: u32 = updates.iter().map(|u| u.num_samples).sum();
        let mut global_loss = 0.0f32;

        // FedAvg: weighted average by sample count
        let mut agg_weights: Vec<f32> = updates[0].weights.clone();
        for w in &mut agg_weights { *w = 0.0; }

        for update in &updates {
            let weight = update.num_samples as f32 / total_samples as f32;
            for (i, w) in update.weights.iter().enumerate() {
                if i < agg_weights.len() {
                    agg_weights[i] += w * weight;
                }
            }
            global_loss += update.local_loss * weight;
        }

        // Store aggregated model
        if let Some(first) = updates.first() {
            self.global_model.insert(first.stitch_type.clone(), agg_weights);
        }

        let result = AggregationResult { global_loss, participating_clients: updates.len() };
        self.audit.add_entry("aggregate", "server", &format!("round={},clients={},loss={:.4}", round_id, result.participating_clients, result.global_loss))
            .map_err(|e| e.to_string())?;

        Ok(result)
    }

    pub fn get_global_model(&self, stitch_type: &str) -> Option<Vec<f32>> {
        self.global_model.get(stitch_type).cloned()
    }

    pub fn current_round(&self) -> u64 { *self.round_number.lock().unwrap() }
}
