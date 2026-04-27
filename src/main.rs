use std::sync::{Arc, Mutex};
use anyhow::Result;
use log::info;
use crate::audit::AuditChain;
use crate::vector_db::VectorDb;
use crate::grpc_service::EmbroideryServiceImpl;
use crate::rest_api::{AppState, create_router};
use crate::web_dashboard::dashboard_html;

pub mod embroidery {
    tonic::include_proto!("embroidery");
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    info!("🧵 Embroidery Agent Server v0.2.0 starting...");

    let audit = Arc::new(AuditChain::new("embroidery_audit.db")?);
    let vector_db = Arc::new(Mutex::new(VectorDb::new(768)));
    let grpc_service = EmbroideryServiceImpl::new(audit.clone(), vector_db.clone());

    // gRPC server (port 50051)
    let grpc_addr = "[::]:50051".parse()?;
    let grpc_svc = embroidery::embroidery_service_server::EmbroideryServiceServer::new(grpc_service);
    tokio::spawn(async move {
        info!("gRPC listening on :50051");
        if let Err(e) = tonic::transport::Server::builder().add_service(grpc_svc).serve(grpc_addr).await {
            eprintln!("gRPC error: {}", e);
        }
    });

    // REST API + Dashboard (port 8080)
    let state = AppState { audit: audit.clone(), vector_db: vector_db.clone(), fed_server: Arc::new(crate::fed_server::FedServer::new(audit.clone(), vector_db.clone())) };
    let app = create_router(state)
        .route("/", axum::routing::get(|| async { dashboard_html() }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!("REST API listening on :8080");
    info!("Dashboard: http://localhost:8080");
    axum::serve(listener, app).await?;

    Ok(())
}
