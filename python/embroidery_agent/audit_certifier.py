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

In production, connects to Rust audit server via gRPC.
For standalone use, uses pure-Python with sqlite3.
"""

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
    """Certificate for an embroidery design."""
    certificate_id: str
    design_hash: str
    designer_id: str
    stitch_count: int
    color_count: int
    created_at: str
    audit_hash: str


class AuditCertifier:
    """Blockchain-style audit chain for embroidery design certification.

    Each entry contains:
        - SHA-256 hash of (index + timestamp + operation + client_id + details + prev_hash)
        - Links to previous entry via prev_hash
        - Tamper-evident: any modification breaks the chain
    """

    def __init__(self, db_path: str = "embroidery_audit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_log (
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
                stitch_count INTEGER,
                color_count INTEGER,
                created_at TEXT NOT NULL,
                audit_hash TEXT NOT NULL
            );
        """)
        conn.close()

    @staticmethod
    def _compute_hash(index: int, timestamp: str, operation: str,
                      client_id: str, details: str, prev_hash: str) -> str:
        """Compute SHA-256 hash for audit entry (mirrors embodied-fl/audit.rs)."""
        data = f"{index}:{timestamp}:{operation}:{client_id}:{details}:{prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def add_entry(self, operation: str, client_id: str, details: str) -> AuditEntry:
        """Add a new entry to the audit chain."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get previous hash
        row = conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = row["hash"] if row else "GENESIS"

        # Get next index
        row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM audit_log").fetchone()
        index = row["next_id"]

        timestamp = datetime.now(timezone.utc).isoformat()
        entry_hash = self._compute_hash(index, timestamp, operation, client_id, details, prev_hash)

        conn.execute(
            "INSERT INTO audit_log (timestamp, operation, client_id, details, hash, prev_hash) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, operation, client_id, details, entry_hash, prev_hash),
        )
        conn.commit()
        conn.close()

        return AuditEntry(index=index, timestamp=timestamp, operation=operation,
                          client_id=client_id, details=details, hash=entry_hash, prev_hash=prev_hash)

    def verify_chain(self) -> Tuple[bool, int]:
        """Verify the integrity of the entire audit chain."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
        conn.close()

        if not rows:
            return True, 0

        for i, row in enumerate(rows):
            entry = AuditEntry(index=row["id"], timestamp=row["timestamp"],
                               operation=row["operation"], client_id=row["client_id"],
                               details=row["details"], hash=row["hash"], prev_hash=row["prev_hash"])
            if i == 0 and entry.prev_hash != "GENESIS":
                return False, len(rows)
            if i > 0:
                expected_prev = rows[i - 1]["hash"]
                if entry.prev_hash != expected_prev:
                    return False, len(rows)
            expected_hash = self._compute_hash(entry.index, entry.timestamp, entry.operation,
                                                entry.client_id, entry.details, entry.prev_hash)
            if entry.hash != expected_hash:
                return False, len(rows)
        return True, len(rows)

    def certify_design(self, design_hash: str, designer_id: str,
                       stitch_count: int, color_count: int) -> DesignCertificate:
        """Issue a design certificate with audit trail."""
        import uuid
        cert_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        entry = self.add_entry("certify_design", designer_id,
                               f"design={design_hash},stitches={stitch_count},colors={color_count}")

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO certificates VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cert_id, design_hash, designer_id, stitch_count, color_count, created_at, entry.hash),
        )
        conn.commit()
        conn.close()

        return DesignCertificate(certificate_id=cert_id, design_hash=design_hash,
                                designer_id=designer_id, stitch_count=stitch_count,
                                color_count=color_count, created_at=created_at, audit_hash=entry.hash)

    def get_recent(self, limit: int = 10, operation_type: Optional[str] = None) -> List[AuditEntry]:
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
        return [AuditEntry(index=r["id"], timestamp=r["timestamp"], operation=r["operation"],
                           client_id=r["client_id"], details=r["details"],
                           hash=r["hash"], prev_hash=r["prev_hash"]) for r in reversed(rows)]

    @property
    def chain_length(self) -> int:
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        conn.close()
        return count
