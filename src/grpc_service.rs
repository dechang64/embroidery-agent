use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};
use crate::audit::AuditChain;
use crate::vector_db::VectorDb;
use crate::fed_server::FedServer;

pub mod embroidery {
    tonic::include_proto!("embroidery");
}

use embroidery::embroidery_service_server::EmbroideryService;
use embroidery::*;

pub struct EmbroideryServiceImpl {
    audit: Arc<AuditChain>,
    vector_db: Arc<Mutex<VectorDb>>,
    fed_server: Arc<FedServer>,
}

impl EmbroideryServiceImpl {
    pub fn new(audit: Arc<AuditChain>, vector_db: Arc<Mutex<VectorDb>>) -> Self {
        let fed = Arc::new(FedServer::new(audit.clone(), vector_db.clone()));
        Self { audit, vector_db, fed_server: fed }
    }
}

#[tonic::async_trait]
impl EmbroideryService for EmbroideryServiceImpl {
    async fn compute_fingerprint(&self, req: Request<FingerprintRequest>) -> Result<Response<FingerprintResponse>, Status> {
        let req = req.into_inner();
        let style_hash = format!("{:x}", md5_placeholder(&req.pattern_id));
        Ok(Response::new(FingerprintResponse {
            pattern_id: req.pattern_id,
            feature_vector: vec![0.0; 768],  // Placeholder — real impl uses DINOv2
            style_hash,
            compute_time_ms: 42,
        }))
    }

    async fn search_patterns(&self, req: Request<SearchRequest>) -> Result<Response<SearchResponse>, Status> {
        let req = req.into_inner();
        let db = self.vector_db.lock().unwrap();
        let results = db.search(&req.query_vector, req.top_k as usize);
        let matches = results.into_iter().map(|(id, sim)| SearchResponse::Match {
            pattern_id: id, name: format!("pattern_{}", id), similarity: sim, stitch_types: vec![],
        }).collect();
        Ok(Response::new(SearchResponse { matches }))
    }

    async fn add_pattern(&self, req: Request<AddPatternRequest>) -> Result<Response<AddPatternResponse>, Status> {
        let req = req.into_inner();
        let mut db = self.vector_db.lock().unwrap();
        db.add_pattern(&req.pattern_id, &req.name, req.feature_vector, req.stitch_types, req.color_count as usize, req.stitch_count as usize);
        Ok(Response::new(AddPatternResponse { success: true }))
    }

    async fn plan_stitches(&self, _req: Request<PlanRequest>) -> Result<Response<PlanResponse>, Status> {
        Ok(Response::new(PlanResponse { success: true, stitch_count: 100, color_count: 5 }))
    }

    async fn export_pattern(&self, _req: Request<ExportRequest>) -> Result<Response<ExportResponse>, Status> {
        Ok(Response::new(ExportResponse { success: true, file_path: "output.dst".to_string(), stitch_count: 100, color_count: 5 }))
    }

    async fn add_audit_entry(&self, req: Request<AuditEntryRequest>) -> Result<Response<AuditEntryResponse>, Status> {
        let req = req.into_inner();
        match self.audit.add_entry(&req.operation, &req.client_id, &req.details) {
            Ok(entry) => Ok(Response::new(AuditEntryResponse { success: true, index: entry.index, hash: entry.hash })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn verify_chain(&self, _req: Request<VerifyRequest>) -> Result<Response<VerifyResponse>, Status> {
        match self.audit.verify_chain() {
            Ok((valid, length)) => Ok(Response::new(VerifyResponse { valid, chain_length: length })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn get_certificate(&self, req: Request<CertRequest>) -> Result<Response<CertResponse>, Status> {
        let req = req.into_inner();
        match self.audit.certify_design(&req.design_hash, &req.designer_id, 100, 5) {
            Ok(cert) => Ok(Response::new(CertResponse { success: true, certificate_id: cert.certificate_id, design_hash: cert.design_hash, created_at: cert.created_at })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn register_client(&self, req: Request<RegisterRequest>) -> Result<Response<RegisterResponse>, Status> {
        let req = req.into_inner();
        match self.fed_server.register_client(&req.client_id, &req.workshop_name, &req.specialty) {
            Ok(()) => Ok(Response::new(RegisterResponse { success: true, client_id: req.client_id })),
            Err(e) => Err(Status::internal(e)),
        }
    }

    async fn submit_update(&self, req: Request<UpdateRequest>) -> Result<Response<UpdateResponse>, Status> {
        let req = req.into_inner();
        let update = crate::fed_server::ClientUpdate {
            client_id: req.client_id, round_id: req.round_id as u64,
            num_samples: req.num_samples as u32, local_loss: req.local_loss,
            weights: req.weights, stitch_type: req.stitch_type,
        };
        match self.fed_server.submit_update(update) {
            Ok(()) => Ok(Response::new(UpdateResponse { accepted: true })),
            Err(e) => Err(Status::internal(e)),
        }
    }

    async fn get_global_model(&self, req: Request<ModelRequest>) -> Result<Response<ModelResponse>, Status> {
        let req = req.into_inner();
        let weights = self.fed_server.get_global_model("satin").unwrap_or_default();
        Ok(Response::new(ModelResponse { round_id: self.fed_server.current_round(), weights, global_loss: 0.0 }))
    }

    async fn start_round(&self, _req: Request<RoundRequest>) -> Result<Response<RoundResponse>, Status> {
        match self.fed_server.start_round() {
            Ok(round_id) => Ok(Response::new(RoundResponse { success: true, round_id })),
            Err(e) => Err(Status::internal(e)),
        }
    }

    async fn aggregate_round(&self, req: Request<AggregateRequest>) -> Result<Response<AggregateResponse>, Status> {
        let req = req.into_inner();
        match self.fed_server.aggregate(req.round_id as u64) {
            Ok(result) => Ok(Response::new(AggregateResponse { success: true, global_loss: result.global_loss, participating_clients: result.participating_clients as i32 })),
            Err(e) => Err(Status::internal(e)),
        }
    }
}

fn md5_placeholder(s: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    format!("{:x}", hasher.finalize())
}
