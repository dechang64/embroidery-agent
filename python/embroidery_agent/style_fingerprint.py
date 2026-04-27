"""
Style Fingerprint module — DINOv2-based embroidery pattern style matching.

Uses our own pure-NumPy HNSW index (from reading-fl/matching/hnsw_index.py)
for pattern library search — zero external dependencies.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json


@dataclass
class PatternRecord:
    """A stored embroidery pattern with its style fingerprint."""
    pattern_id: str
    name: str
    feature_vector: np.ndarray  # 768-dim DINOv2 feature
    stitch_types: List[str] = field(default_factory=list)
    color_count: int = 0
    stitch_count: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternLibrary:
    """Pattern library backed by our own pure-NumPy HNSW index.

    Uses reading-fl/matching/hnsw_index.py — zero external dependencies,
    pure NumPy HNSW with Euclidean distance. No hnsw crate, no space crate.
    """

    def __init__(self, dimension: int = 768, persist_path: Optional[str] = None):
        from .hnsw_index import HNSWIndex
        self.dimension = dimension
        self._patterns: Dict[str, PatternRecord] = {}
        self._id_map: Dict[int, str] = {}  # HNSW idx → pattern_id
        self._next_idx = 0
        self._index = HNSWIndex(dim=dimension, M=16, ef_construction=200, ef_search=50)
        self.persist_path = persist_path
        if persist_path and Path(persist_path).exists():
            self._load()

    def add(self, pattern: PatternRecord) -> str:
        """Add a pattern to the library via HNSW index."""
        assert pattern.feature_vector.shape == (self.dimension,), \
            f"Expected ({self.dimension},), got {pattern.feature_vector.shape}"
        pid = pattern.pattern_id
        self._patterns[pid] = pattern
        hnsw_idx = self._next_idx
        self._id_map[hnsw_idx] = pid
        self._next_idx += 1
        self._index.add(
            pattern.feature_vector,
            idx=hnsw_idx,
            metadata={"pattern_id": pid, "name": pattern.name,
                      "stitch_types": ",".join(pattern.stitch_types)},
        )
        if self.persist_path:
            self._save()
        return pid

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar patterns using HNSW approximate nearest neighbor."""
        assert query.shape == (self.dimension,), f"Expected ({self.dimension},), got {query.shape}"
        results = self._index.search(query, k=top_k)
        # HNSW returns (distance, idx, metadata) tuples
        output = []
        for r in results:
            hnsw_idx = r[1]
            pid = self._id_map.get(hnsw_idx)
            if pid and pid in self._patterns:
                output.append((pid, r[0]))
        return output

    def remove(self, pattern_id: str) -> bool:
        """Remove a pattern from the library."""
        if pattern_id not in self._patterns:
            return False
        hnsw_idx = None
        for idx, pid in self._id_map.items():
            if pid == pattern_id:
                hnsw_idx = idx
                break
        if hnsw_idx is not None:
            self._index.remove(hnsw_idx)
            del self._id_map[hnsw_idx]
        del self._patterns[pattern_id]
        if self.persist_path:
            self._save()
        return True

    def get(self, pattern_id: str) -> Optional[PatternRecord]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def __len__(self) -> int:
        return len(self._patterns)

    def _save(self):
        """Persist library to disk."""
        data = {"dimension": self.dimension, "patterns": {}}
        for pid, p in self._patterns.items():
            data["patterns"][pid] = {
                "pattern_id": p.pattern_id, "name": p.name,
                "feature_vector": p.feature_vector.tolist(),
                "stitch_types": p.stitch_types, "color_count": p.color_count,
                "stitch_count": p.stitch_count, "created_at": p.created_at,
                "metadata": p.metadata,
            }
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        """Load library from disk and rebuild HNSW index."""
        with open(self.persist_path) as f:
            data = json.load(f)
        self.dimension = data["dimension"]
        for pid, pd in data["patterns"].items():
            self._patterns[pid] = PatternRecord(
                pattern_id=pd["pattern_id"], name=pd["name"],
                feature_vector=np.array(pd["feature_vector"], dtype=np.float32),
                stitch_types=pd.get("stitch_types", []),
                color_count=pd.get("color_count", 0),
                stitch_count=pd.get("stitch_count", 0),
                created_at=pd.get("created_at", ""),
                metadata=pd.get("metadata", {}),
            )
            hnsw_idx = self._next_idx
            self._id_map[hnsw_idx] = pid
            self._next_idx += 1
            self._index.add(
                self._patterns[pid].feature_vector, idx=hnsw_idx,
                metadata={"pattern_id": pid, "name": pd["name"]},
            )


class StyleFingerprint:
    """Computes style fingerprints for embroidery patterns.

    In production, uses DINOv2 (facebook/dinov2-base) for 768-dim features.
    Falls back to hash-based fingerprinting when torch is unavailable.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", use_torch: bool = True):
        self.model_name = model_name
        self.use_torch = use_torch and self._check_torch()
        self._model = None

    def _check_torch(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    def compute(self, image) -> np.ndarray:
        """Compute 768-dim style fingerprint for an image."""
        if self.use_torch:
            return self._compute_dinov2(image)
        return self._compute_hash_fallback(image)

    def _compute_dinov2(self, image) -> np.ndarray:
        """DINOv2 feature extraction."""
        import torch
        from PIL import Image
        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self._model.eval()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img = image.convert("RGB").resize((224, 224))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self._model(img_tensor)

        return features.squeeze().numpy().astype(np.float32)

    def _compute_hash_fallback(self, image) -> np.ndarray:
        """Hash-based fallback when torch is unavailable."""
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img = image.convert("RGB").resize((28, 28))
        pixels = np.array(img).flatten().astype(np.float32) / 255.0
        fingerprint = np.zeros(768, dtype=np.float32)
        n = min(len(pixels), 768)
        fingerprint[:n] = pixels[:n]
        norm = np.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint /= norm
        return fingerprint

    @staticmethod
    def style_hash(feature_vector: np.ndarray) -> str:
        """Generate a deterministic style hash from feature vector."""
        h = hashlib.sha256(feature_vector.tobytes()).hexdigest()
        return h[:16]
