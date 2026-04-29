"""Tests for embodied-fl integration modules."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from embroidery_agent.style_fingerprint import StyleFingerprint, PatternLibrary, PatternRecord
from embroidery_agent.audit_certifier import AuditCertifier, DesignCertificate


# ============================================================
# StyleFingerprint Tests
# ============================================================

class TestStyleFingerprint:
    def test_compute_hash_deterministic(self):
        fp = StyleFingerprint()
        vec = np.random.randn(768).astype(np.float32)
        h1 = fp.compute_hash(vec)
        h2 = fp.compute_hash(vec)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_hash_different_vectors(self):
        fp = StyleFingerprint()
        v1 = np.random.randn(768).astype(np.float32)
        v2 = np.random.randn(768).astype(np.float32)
        assert fp.compute_hash(v1) != fp.compute_hash(v2)

    def test_cosine_similarity_identical(self):
        fp = StyleFingerprint()
        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        sim = fp.compute_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        fp = StyleFingerprint()
        v1 = np.zeros(768, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(768, dtype=np.float32)
        v2[1] = 1.0
        sim = fp.compute_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_extract_from_image(self, simple_image):
        fp = StyleFingerprint()
        # Save temp image for extract
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            simple_image.save(f.name)
            feature = fp.extract(f.name)
            os.unlink(f.name)
        assert feature.shape == (768,)
        assert feature.dtype == np.float32


# ============================================================
# PatternLibrary Tests
# ============================================================

class TestPatternLibrary:
    def test_add_and_search(self):
        lib = PatternLibrary()
        v1 = np.random.randn(768).astype(np.float32)
        v2 = np.random.randn(768).astype(np.float32)

        lib.add(PatternRecord("p1", "Pattern 1", v1, stitch_types=["fill"]))
        lib.add(PatternRecord("p2", "Pattern 2", v2, stitch_types=["satin"]))

        assert lib.size == 2

        results = lib.search(v1, k=1)
        assert len(results) == 1
        assert results[0][0] == "p1"
        assert results[0][1] > 0.99  # identical vector

    def test_search_returns_sorted(self):
        lib = PatternLibrary()
        query = np.random.randn(768).astype(np.float32)

        # Add patterns with varying similarity
        for i in range(5):
            noise = np.random.randn(768).astype(np.float32) * (i * 0.5)
            vec = query + noise
            vec /= np.linalg.norm(vec)
            lib.add(PatternRecord(f"p{i}", f"Pattern {i}", vec))

        results = lib.search(query, k=5)
        # First result should be most similar
        assert results[0][1] >= results[-1][1]

    def test_delete_pattern(self):
        lib = PatternLibrary()
        vec = np.random.randn(768).astype(np.float32)
        lib.add(PatternRecord("p1", "Pattern 1", vec))
        assert lib.size == 1

        lib.delete("p1")
        assert lib.size == 0

    def test_persistence(self, tmp_path):
        db_path = str(tmp_path / "patterns.json")
        lib = PatternLibrary(persist_path=db_path)
        vec = np.random.randn(768).astype(np.float32)
        lib.add(PatternRecord("p1", "Test Pattern", vec, stitch_types=["fill", "running"]))
        lib._save()

        # Reload
        lib2 = PatternLibrary(persist_path=db_path)
        lib2._load()
        assert lib2.size == 1
        results = lib2.search(vec, k=1)
        assert results[0][0] == "p1"

    def test_empty_search(self):
        lib = PatternLibrary()
        vec = np.random.randn(768).astype(np.float32)
        results = lib.search(vec, k=5)
        assert results == []


# ============================================================
# AuditCertifier Tests
# ============================================================

class TestAuditCertifier:
    def test_append_and_verify(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        entry = certifier.append("test_operation", "client_1", "test details")
        assert entry.index == 1
        assert entry.prev_hash == "GENESIS"
        assert len(entry.hash) == 64

        valid, length, latest = certifier.verify_chain()
        assert valid
        assert length == 1
        assert latest == entry.hash

    def test_chain_grows(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        for i in range(5):
            certifier.append(f"op_{i}", f"client_{i % 2}", f"details_{i}")

        assert certifier.chain_length == 5

        valid, length, _ = certifier.verify_chain()
        assert valid
        assert length == 5

    def test_certify_design(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        cert = certifier.certify_design(
            design_id="flower_001",
            designer_id="designer_alice",
            stitch_count=5000,
            color_count=4,
            file_formats=["pes", "dst"],
        )

        assert cert.design_id == "flower_001"
        assert cert.designer_id == "designer_alice"
        assert cert.stitch_count == 5000
        assert len(cert.audit_hash) == 64
        assert cert.prev_audit_hash == "GENESIS"

    def test_certificate_verify(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        cert = certifier.certify_design(
            design_id="test_001",
            designer_id="bob",
            stitch_count=1000,
            color_count=2,
            file_formats=["pes"],
        )

        assert cert.verify("GENESIS")

    def test_certificate_tamper_detect(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        cert = certifier.certify_design(
            design_id="secure_001",
            designer_id="eve",
            stitch_count=9999,
            color_count=5,
            file_formats=["pes", "dst", "svg"],
        )

        # Tamper with stitch count
        cert.stitch_count = 1
        assert not cert.verify("GENESIS")

    def test_get_recent(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        for i in range(10):
            certifier.append(f"op_{i}", "client_1", f"detail_{i}")

        recent = certifier.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[0].index == 8  # 8, 9, 10 (reversed)

    def test_filter_by_operation(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        certifier.append("design_certify", "alice", "cert1")
        certifier.append("design_export", "alice", "export1")
        certifier.append("design_certify", "bob", "cert2")

        certs = certifier.get_recent(operation_type="design_certify")
        assert len(certs) == 2

    def test_chain_tamper_detect(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        certifier = AuditCertifier(db_path=db_path)

        certifier.append("op1", "c1", "d1")
        certifier.append("op2", "c2", "d2")

        # Directly tamper with database
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE audit_log SET details = 'TAMPERED' WHERE id = 1")
        conn.commit()
        conn.close()

        valid, length, broken_hash = certifier.verify_chain()
        assert not valid
        assert length == 2


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def simple_image():
    """Create a simple test image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(255, 0, 0))
    return img
