use std::collections::HashMap;
use sha2::{Sha256, Digest};
use rusqlite::Connection;
use chrono::Utc;
use anyhow::Result;

/// Blockchain-style audit chain for embroidery design certification.
/// Mirrors embodied-fl/audit.rs — SHA-256 hash chain with SQLite persistence.
pub struct AuditChain {
    conn: Connection,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditEntry {
    pub index: i64,
    pub timestamp: String,
    pub operation: String,
    pub client_id: String,
    pub details: String,
    pub hash: String,
    pub prev_hash: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DesignCertificate {
    pub certificate_id: String,
    pub design_hash: String,
    pub designer_id: String,
    pub stitch_count: i64,
    pub color_count: i64,
    pub created_at: String,
    pub audit_hash: String,
}

impl AuditChain {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                client_id TEXT NOT NULL,
                details TEXT NOT NULL,
                hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS certificates (
                certificate_id TEXT PRIMARY KEY,
                design_hash TEXT NOT NULL,
                designer_id TEXT NOT NULL,
                stitch_count INTEGER DEFAULT 0,
                color_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                audit_hash TEXT NOT NULL
            );"
        )?;
        Ok(Self { conn })
    }

    fn compute_hash(index: i64, timestamp: &str, operation: &str,
                    client_id: &str, details: &str, prev_hash: &str) -> String {
        let input = format!("{}:{}:{}:{}:{}:{}", index, timestamp, operation, client_id, details, prev_hash);
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn get_latest_hash(&self) -> Result<String> {
        let hash: String = self.conn.query_row(
            "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0),
        ).unwrap_or_else(|_| "GENESIS".to_string());
        Ok(hash)
    }

    pub fn add_entry(&self, operation: &str, client_id: &str, details: &str) -> Result<AuditEntry> {
        let prev_hash = self.get_latest_hash()?;
        let index: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM audit_log", [], |row| row.get(0)
        )?;
        let timestamp = Utc::now().to_rfc3339();
        let hash = Self::compute_hash(index, &timestamp, operation, client_id, details, &prev_hash);

        self.conn.execute(
            "INSERT INTO audit_log (timestamp, operation, client_id, details, hash, prev_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            [&timestamp, operation, client_id, details, &hash, &prev_hash],
        )?;

        Ok(AuditEntry { index, timestamp, operation: operation.to_string(), client_id: client_id.to_string(), details: details.to_string(), hash, prev_hash })
    }

    pub fn verify_chain(&self) -> Result<(bool, i64)> {
        let mut stmt = self.conn.prepare("SELECT id, timestamp, operation, client_id, details, hash, prev_hash FROM audit_log ORDER BY id ASC")?;
        let entries: Vec<AuditEntry> = stmt.query_map([], |row| {
            Ok(AuditEntry {
                index: row.get(0)?, timestamp: row.get(1)?, operation: row.get(2)?,
                client_id: row.get(3)?, details: row.get(4)?, hash: row.get(5)?, prev_hash: row.get(6)?,
            })
        })?.filter_map(|r| r.ok()).collect();

        let count = entries.len() as i64;
        if entries.is_empty() { return Ok((true, 0)); }

        for (i, entry) in entries.iter().enumerate() {
            let expected_prev = if i == 0 { "GENESIS".to_string() } else { entries[i - 1].hash.clone() };
            if entry.prev_hash != expected_prev { return Ok((false, count)); }
            let expected_hash = Self::compute_hash(entry.index, &entry.timestamp, &entry.operation, &entry.client_id, &entry.details, &entry.prev_hash);
            if entry.hash != expected_hash { return Ok((false, count)); }
        }
        Ok((true, count))
    }

    pub fn certify_design(&self, design_hash: &str, designer_id: &str, stitch_count: i64, color_count: i64) -> Result<DesignCertificate> {
        let cert_id = uuid::Uuid::new_v4().to_string();
        let created_at = Utc::now().to_rfc3339();
        let entry = self.add_entry("certify_design", designer_id, &format!("design={},stitches={},colors={}", design_hash, stitch_count, color_count))?;

        self.conn.execute(
            "INSERT INTO certificates (certificate_id, design_hash, designer_id, stitch_count, color_count, created_at, audit_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            [&cert_id, design_hash, designer_id, &stitch_count.to_string(), &color_count.to_string(), &created_at, &entry.hash],
        )?;

        Ok(DesignCertificate { certificate_id: cert_id, design_hash: design_hash.to_string(), designer_id: designer_id.to_string(), stitch_count, color_count, created_at, audit_hash: entry.hash })
    }

    pub fn chain_length(&self) -> Result<i64> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM audit_log", [], |row| row.get(0))?;
        Ok(count)
    }
}
