"""
Audit Certifier — blockchain-style audit chain for embroidery design certification.

Ported from embodied-fl/audit.rs:
    - SHA-256 hash chain (same algorithm as embodied-fl AuditChain)
    - SQLite persistence (same schema structure)
    - Chain verification (same verify_chain logic)

Extended for embroidery:
    - Design certification (design_hash + stitch_count + color_count)
    - Copyright proof (designer_id + timestamp + hash)
    - Workshop audit trail (multi-workshop collaboration tracking)

In production, this connects to the Rust audit server via gRPC.
For MVP, uses pure-Python with sqlite3.
"""

from __future__ import annotations
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AuditEntry:
    """Single audit chain entry (mirrors embodied-fl AuditEntry)."""
    index: int
    timestamp: str
    operation: str
    client_id: str
    details: str
    hash: str
    prev_hash: str


@dataclass
class DesignCertificate:
    """Certificate for an embroidery design (audit chain proof)."""
    design_id: str
    design_hash: str
    designer_id: str
    stitch_count: int
    color_count: int
    file_formats: List[str]
    created_at: str
    audit_hash: str
    audit_index: int
    prev_audit_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def verify(self, expected_prev_hash: Optional[str] = None) -> bool:
        """Verify certificate integrity.

        Args:
            expected_prev_hash: If provided, also verify chain linkage.
        """
        # Reconstruct hash
        data = (
            f"{self.design_id}:{self.design_hash}:{self.designer_id}:"
            f"{self.stitch_count}:{self.color_count}:{self.created_at}:"
            f"{self.prev_audit_hash}"
        )
        computed = hashlib.sha256(data.encode()).hexdigest()
        hash_ok = computed == self.audit_hash
        if expected_prev_hash is not None:
            return hash_ok and self.prev_audit_hash == expected_prev_hash
        return hash_ok


class AuditCertifier:
    """Blockchain-style audit chain for embroidery design certification.

    Architecture mirrors embodied-fl/audit.rs:
        - SQLite WAL mode for concurrent access
        - SHA-256 hash chain for tamper-proof records
        - Chain verification for integrity checks

    Operations:
        - design_create: New design registered
        - design_export: Design exported to machine format
        - design_modify: Design modified
        - workshop_collab: Multi-workshop collaboration event
    """

    def __init__(self, db_path: str = "embroidery_audit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database (mirrors embodied-fl AuditChain::new)."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                client_id TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT '',
                hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_client ON audit_log(client_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        conn.commit()
        conn.close()

    def _compute_hash(self, index: int, timestamp: str, operation: str,
                      client_id: str, details: str, prev_hash: str) -> str:
        """Compute SHA-256 hash (same algorithm as embodied-fl audit.rs)."""
        data = f"{index}:{timestamp}:{operation}:{client_id}:{details}:{prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def append(self, operation: str, client_id: str = "",
               details: str = "") -> AuditEntry:
        """Append an audit entry (mirrors embodied-fl AuditChain::append).

        Also syncs to FedCtx audit service if available.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get previous hash
        row = conn.execute(
            "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = row["hash"] if row else "GENESIS"

        # Get next index
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM audit_log"
        ).fetchone()
        index = row[0]

        timestamp = datetime.now(timezone.utc).isoformat()
        hash_val = self._compute_hash(index, timestamp, operation, client_id, details, prev_hash)

        conn.execute(
            "INSERT INTO audit_log (timestamp, operation, client_id, details, hash, prev_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, operation, client_id, details, hash_val, prev_hash),
        )
        conn.commit()
        conn.close()

        entry = AuditEntry(
            index=index, timestamp=timestamp, operation=operation,
            client_id=client_id, details=details,
            hash=hash_val, prev_hash=prev_hash,
        )

        # Sync to FedCtx audit service
        try:
            from core.grpc_client import get_fedctx_client
            fedctx = get_fedctx_client()
            if fedctx.available:
                fedctx.audit_append(
                    event_type=operation,
                    node_id=client_id or "embroidery_server",
                    metadata={"details": details, "hash": hash_val},
                )
        except (ImportError, Exception):
            pass

        return entry

    def certify_design(self, design_id: str, designer_id: str,
                       stitch_count: int, color_count: int,
                       file_formats: List[str],
                       design_hash: str = "") -> DesignCertificate:
        """Create a design certificate on the audit chain."""
        if not design_hash:
            design_hash = hashlib.sha256(
                f"{design_id}:{stitch_count}:{color_count}:{time.time()}".encode()
            ).hexdigest()[:16]

        # Get previous audit hash
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = row[0] if row else "GENESIS"
        conn.close()

        created_at = datetime.now(timezone.utc).isoformat()

        # Compute certificate hash
        cert_data = (
            f"{design_id}:{design_hash}:{designer_id}:"
            f"{stitch_count}:{color_count}:{created_at}:{prev_hash}"
        )
        audit_hash = hashlib.sha256(cert_data.encode()).hexdigest()

        # Get audit index
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM audit_log"
        ).fetchone()
        audit_index = row[0]
        conn.close()

        # Append to audit chain
        self.append(
            operation="design_certify",
            client_id=designer_id,
            details=json.dumps({
                "design_id": design_id,
                "design_hash": design_hash,
                "stitch_count": stitch_count,
                "color_count": color_count,
                "file_formats": file_formats,
            }),
        )

        return DesignCertificate(
            design_id=design_id,
            design_hash=design_hash,
            designer_id=designer_id,
            stitch_count=stitch_count,
            color_count=color_count,
            file_formats=file_formats,
            created_at=created_at,
            audit_hash=audit_hash,
            audit_index=audit_index,
            prev_audit_hash=prev_hash,
        )

    def verify_chain(self) -> Tuple[bool, int, str]:
        """Verify audit chain integrity (mirrors embodied-fl AuditChain::verify_chain)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        rows = conn.execute(
            "SELECT id, timestamp, operation, client_id, details, hash, prev_hash "
            "FROM audit_log ORDER BY id ASC"
        ).fetchall()
        conn.close()

        prev_hash = "GENESIS"
        for row in rows:
            if row["prev_hash"] != prev_hash:
                return (False, count, row["hash"])

            expected = self._compute_hash(
                row["id"], row["timestamp"], row["operation"],
                row["client_id"], row["details"], row["prev_hash"],
            )
            if row["hash"] != expected:
                return (False, count, row["hash"])

            prev_hash = row["hash"]

        latest = rows[-1]["hash"] if rows else ""
        return (True, count, latest)

    def get_recent(self, limit: int = 10,
                   operation_type: Optional[str] = None) -> List[AuditEntry]:
        """Get recent audit entries."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        sql = "SELECT * FROM audit_log"
        params = []
        if operation_type:
            sql += " WHERE operation = ?"
            params.append(operation_type)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        conn.close()

        entries = [
            AuditEntry(
                index=r["id"], timestamp=r["timestamp"],
                operation=r["operation"], client_id=r["client_id"],
                details=r["details"], hash=r["hash"], prev_hash=r["prev_hash"],
            )
            for r in reversed(rows)
        ]
        return entries

    @property
    def chain_length(self) -> int:
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        conn.close()
        return count
