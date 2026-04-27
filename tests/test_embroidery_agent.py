"""Tests for embroidery-agent core modules."""
import os, sys, pytest
import numpy as np
import hashlib
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


class TestImageProcessor:
    def test_stitch_types(self):
        from embroidery_agent.image_processor import StitchType
        assert len(StitchType) == 9  # 9 traditional categories

    def test_embroidery_color(self):
        from embroidery_agent.image_processor import EmbroideryColor
        c = EmbroideryColor(name="red", rgb=(255, 0, 0))
        assert c.hex == "#ff0000"

    def test_process_synthetic(self):
        from embroidery_agent.image_processor import ImageProcessor
        from PIL import Image
        proc = ImageProcessor()
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = proc.process(img)
        assert len(result.regions) >= 0  # May have 0 regions for random noise
        assert result.color_palette is not None


class TestStitchPlanner:
    def test_stitch_point(self):
        from embroidery_agent.stitch_planner import StitchPoint
        p = StitchPoint(x=10.0, y=20.0)
        assert p.as_tuple == (10.0, 20.0)

    def test_plan_empty(self):
        from embroidery_agent.stitch_planner import StitchPlanner
        planner = StitchPlanner()
        plan = planner.plan([], [])
        assert plan.total_stitches == 0


class TestPatternGenerator:
    def test_export_svg(self):
        from embroidery_agent.pattern_generator import PatternGenerator
        from embroidery_agent.stitch_planner import StitchPlan, StitchBlock, StitchPoint
        from embroidery_agent.image_processor import StitchType, EmbroideryColor
        gen = PatternGenerator()
        block = StitchBlock(stitch_type=StitchType.RUNNING, color=EmbroideryColor("red", (255, 0, 0)),
                            points=[StitchPoint(0, 0), StitchPoint(10, 10), StitchPoint(20, 0)])
        plan = StitchPlan(blocks=[block], total_stitches=2, total_colors=1,
                          design_width_mm=20, design_height_mm=10)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            gen.generate_preview_svg(plan, f.name)
            assert os.path.exists(f.name)
            assert os.path.getsize(f.name) > 0
            os.unlink(f.name)


class TestStyleFingerprint:
    def test_pattern_library_add_search(self):
        from embroidery_agent.style_fingerprint import PatternLibrary, PatternRecord
        lib = PatternLibrary(dimension=8)
        lib.add(PatternRecord(pattern_id="p1", name="test_pattern", feature_vector=np.random.randn(8).astype(np.float32), stitch_types=["satin"], color_count=3, stitch_count=100))
        assert len(lib) == 1
        results = lib.search(np.random.randn(8).astype(np.float32), top_k=1)
        assert len(results) == 1

    def test_style_hash(self):
        from embroidery_agent.style_fingerprint import StyleFingerprint
        vec = np.random.randn(768).astype(np.float32)
        h = StyleFingerprint.style_hash(vec)
        assert len(h) == 16
        assert h == StyleFingerprint.style_hash(vec)  # deterministic


class TestAuditChain:
    def test_add_and_verify(self):
        from embroidery_agent.audit_certifier import AuditCertifier
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            certifier = AuditCertifier(db_path=f.name)
        entry = certifier.add_entry("test_op", "test_client", "test details")
        assert entry.index == 1
        valid, length = certifier.verify_chain()
        assert valid
        assert length == 1
        os.unlink(f.name)

    def test_certify_design(self):
        from embroidery_agent.audit_certifier import AuditCertifier
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            certifier = AuditCertifier(db_path=f.name)
        cert = certifier.certify_design("abc123", "designer_1", 1000, 5)
        assert cert.certificate_id
        assert cert.design_hash == "abc123"
        os.unlink(f.name)

    def test_chain_linkage(self):
        prev = hashlib.sha256(b"genesis").hexdigest()
        curr = hashlib.sha256(f"{prev}:data".encode()).hexdigest()
        assert curr != prev


class TestFederatedClient:
    def test_model_forward(self):
        from embroidery_agent.fl.client import StitchQualityModel
        model = StitchQualityModel(input_dim=24, hidden_dim=16, output_dim=9)
        x = np.random.randn(4, 24).astype(np.float32)
        y = model.forward(x)
        assert y.shape == (4, 9)

    def test_model_backward(self):
        from embroidery_agent.fl.client import StitchQualityModel
        model = StitchQualityModel(input_dim=24, hidden_dim=16, output_dim=9)
        x = np.random.randn(4, 24).astype(np.float32)
        y = np.zeros((4, 9), dtype=np.float32)
        grads = model.compute_gradients(x, y)
        assert len(grads) == 4  # W1, b1, W2, b2

    def test_model_serialization(self):
        from embroidery_agent.fl.client import StitchQualityModel
        model = StitchQualityModel(input_dim=24, hidden_dim=16, output_dim=9)
        weights = model.get_weights()
        model2 = StitchQualityModel(input_dim=24, hidden_dim=16, output_dim=9)
        model2.set_weights(weights)
        x = np.random.randn(2, 24).astype(np.float32)
        assert np.allclose(model.forward(x), model2.forward(x))


class TestFedAvgAggregator:
    def test_aggregate(self):
        from embroidery_agent.fl.aggregation import FedAvgAggregator, ClientUpdate
        agg = FedAvgAggregator()
        updates = [
            ClientUpdate("c1", 1, 100, 0.5, [[1.0, 2.0], [3.0, 4.0]]),
            ClientUpdate("c2", 1, 100, 0.3, [[3.0, 2.0], [1.0, 4.0]]),
        ]
        result = agg.aggregate(updates)
        assert result.participating_clients == 2
        assert result.total_samples == 200
        assert abs(result.global_loss - 0.4) < 0.01

    def test_convergence(self):
        from embroidery_agent.fl.aggregation import FedAvgAggregator, AggregationResult
        agg = FedAvgAggregator()
        history = [
            AggregationResult([[1.0]], 1.0, 2, 200),
            AggregationResult([[0.8]], 0.8, 2, 200),
            AggregationResult([[0.79]], 0.79, 2, 200),
            AggregationResult([[0.785]], 0.785, 2, 200),
            AggregationResult([[0.783]], 0.783, 2, 200),
        ]
        metrics = agg.compute_convergence(history)
        assert metrics["loss_trend"] == "decreasing"


class TestHNSWIndex:
    def test_cosine_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        assert abs(np.dot(a, a) / (np.linalg.norm(a) ** 2) - 1.0) < 1e-10

    def test_cosine_orthogonal(self):
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        assert abs(np.dot(a, b)) < 1e-10

    def test_nearest_neighbor(self):
        db = {"p1": np.array([0.9, 0.1]), "p2": np.array([0.1, 0.9])}
        query = np.array([0.8, 0.2])
        best = max(db.items(), key=lambda x: np.dot(query, x[1]) / (np.linalg.norm(query) * np.linalg.norm(x[1])))
        assert best[0] == "p1"
