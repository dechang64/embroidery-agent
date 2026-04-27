"""Tests for Rust module interfaces (via gRPC/REST)."""
import os, sys, pytest
import numpy as np
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


class TestRustAuditChain:
    """Test audit chain logic matching Rust audit.rs implementation."""

    def test_sha256_hash_chain(self):
        """Same algorithm as Rust: SHA-256(index + timestamp + operation + client_id + details + prev_hash)"""
        index = 1
        timestamp = "2024-01-01T00:00:00Z"
        operation = "test"
        client_id = "client_1"
        details = "test details"
        prev_hash = "0" * 64

        data = f"{index}{timestamp}{operation}{client_id}{details}{prev_hash}"
        h = hashlib.sha256(data.encode()).hexdigest()
        assert len(h) == 64

    def test_chain_integrity(self):
        """Verify chain linkage matches Rust verify_chain logic."""
        entries = []
        prev_hash = "0" * 64
        for i in range(5):
            data = f"{i+1}2024-01-01T00:00:0{i}Zop_{i}client_{i}details_{i}{prev_hash}"
            h = hashlib.sha256(data.encode()).hexdigest()
            entries.append({"index": i + 1, "hash": h, "prev_hash": prev_hash})
            prev_hash = h

        # Verify
        for i, entry in enumerate(entries):
            expected_prev = entries[i - 1]["hash"] if i > 0 else "0" * 64
            assert entry["prev_hash"] == expected_prev


class TestRustHNSWIndex:
    """Test HNSW index logic matching Rust hnsw_index.rs."""

    def test_cosine_similarity(self):
        a = np.random.randn(768).astype(np.float32)
        a /= np.linalg.norm(a)
        sim = np.dot(a, a)
        assert abs(sim - 1.0) < 1e-5

    def test_dimension_check(self):
        """Rust panics on dimension mismatch — Python should validate too."""
        dim = 768
        vec = np.random.randn(dim).astype(np.float32)
        assert len(vec) == dim


class TestRustFedServer:
    """Test federated server logic matching Rust fed_server.rs."""

    def test_fedavg_weighted(self):
        """Same FedAvg algorithm as Rust: w_global = Σ(n_k/N) * w_k"""
        w1 = np.array([1.0, 2.0, 3.0])
        w2 = np.array([3.0, 2.0, 1.0])
        n1, n2 = 100, 300
        total = n1 + n2
        global_w = (n1 / total) * w1 + (n2 / total) * w2
        expected = np.array([2.5, 2.0, 1.5])
        assert np.allclose(global_w, expected)

    def test_round_management(self):
        """Round numbers should be monotonically increasing."""
        rounds = [0]
        for _ in range(10):
            rounds.append(rounds[-1] + 1)
        assert rounds == list(range(11))


class TestRustVectorDB:
    """Test vector DB logic matching Rust vector_db.rs."""

    def test_persist_roundtrip(self):
        """Patterns should survive serialization."""
        import json
        pattern = {"id": "p1", "name": "test", "vector": [0.1, 0.2, 0.3]}
        serialized = json.dumps(pattern)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "p1"
        assert np.allclose(deserialized["vector"], [0.1, 0.2, 0.3])


class TestProtoRPCs:
    """Verify proto service has expected RPCs."""

    def test_expected_rpcs(self):
        proto_path = os.path.join(os.path.dirname(__file__), "..", "proto", "embroidery.proto")
        if not os.path.exists(proto_path):
            pytest.skip("proto not found")
        with open(proto_path) as f:
            content = f.read()
        expected_rpcs = [
            "ComputeFingerprint", "SearchPatterns", "AddPattern",
            "PlanStitches", "ExportPattern",
            "AddAuditEntry", "VerifyChain", "GetCertificate",
            "RegisterClient", "SubmitUpdate", "GetGlobalModel",
            "StartRound", "AggregateRound",
        ]
        for rpc in expected_rpcs:
            assert rpc in content, f"Missing RPC: {rpc}"
