"""
Style Fingerprint module — DINOv2-based embroidery pattern style matching.

Ported from embodied-fl/feature_extractor.py:
    - DINOv2 768-dim feature extraction for style fingerprinting
    - HNSW vector index for pattern library search (mirrors embodied-fl/hnsw_index.rs)
    - Pattern library management (add, search, delete)

Use cases:
    - Given a new design, find the most similar existing embroidery patterns
    - Suggest stitch types based on similar historical patterns
    - Style consistency checking across a collection
    - Multi-workshop pattern sharing (federated pattern library)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import time


@dataclass
class PatternRecord:
    """A stored embroidery pattern with its style fingerprint."""
    pattern_id: str
    name: str
    feature_vector: np.ndarray    # 768-dim DINOv2 feature
    stitch_types: List[str] = field(default_factory=list)
    color_count: int = 0
    stitch_count: int = 0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_hash(self) -> str:
        """SHA-256 hash of the feature vector for integrity verification."""
        return hashlib.sha256(self.feature_vector.tobytes()).hexdigest()[:16]


class StyleFingerprint:
    """DINOv2-based style fingerprint extractor for embroidery patterns.

    Extracts 768-dimensional style features from embroidery design images.
    These features capture visual style characteristics (color distribution,
    texture patterns, complexity) without requiring any labels.

    Architecture mirrors embodied-fl's DINOv2SceneExtractor:
        - Input: PIL Image (any size, auto-resized to 518×518)
        - Model: DINOv2 ViT-B/14 (768-dim output)
        - Output: L2-normalized 768-dim feature vector
    """

    DIMENSION = 768
    INPUT_SIZE = 518  # DINOv2 optimal input size

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cpu"):
        """
        Args:
            model_name: DINOv2 model variant
            device: torch device
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._transform = None

    def _ensure_model(self):
        """Lazy-load DINOv2 model (same pattern as embodied-fl)."""
        if self._model is not None:
            return

        import torch
        from torchvision import transforms

        self._model = torch.hub.load(
            "facebookresearch/dinov2", self.model_name
        ).to(self.device).eval()

        self._transform = transforms.Compose([
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def extract(self, image) -> np.ndarray:
        """Extract 768-dim style fingerprint from an image.

        Args:
            image: PIL Image or file path

        Returns:
            L2-normalized 768-dim numpy array
        """
        import torch
        from PIL import Image as PILImage

        self._ensure_model()

        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        elif not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image).convert("RGB")

        tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.forward_features(tensor)
            # Global average pooling on CLS token + patch tokens
            feat = features["x_norm_clstoken"]
            if feat is None:
                feat = features["x_norm_patchtokens"].mean(dim=1)

        # L2 normalize
        feat = feat.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat.astype(np.float32)

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity between two feature vectors."""
        return float(np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
        ))

    def compute_hash(self, feature_vector: np.ndarray) -> str:
        """Compute SHA-256 hash of a feature vector."""
        return hashlib.sha256(feature_vector.tobytes()).hexdigest()


class PatternLibrary:
    """HNSW-based embroidery pattern library for style search.

    Mirrors embodied-fl's HNSW index (hnsw_index.rs) + VectorDB (vector_db.rs).
    Stores DINOv2 style fingerprints and enables fast similarity search.

    In production, this would connect to the Rust HNSW server via gRPC.
    For MVP, uses a pure-Python brute-force implementation.
    """

    def __init__(self, dimension: int = 768, persist_path: Optional[str] = None):
        """
        Args:
            dimension: Feature vector dimension (768 for DINOv2 ViT-B/14)
            persist_path: Optional file path for persistence
        """
        self.dimension = dimension
        self.persist_path = persist_path
        self._patterns: Dict[str, PatternRecord] = {}
        self._vectors: Dict[str, np.ndarray] = {}

        if persist_path and Path(persist_path).exists():
            self._load()

    def add(self, pattern: PatternRecord) -> str:
        """Add a pattern to the library."""
        assert pattern.feature_vector.shape[0] == self.dimension
        self._patterns[pattern.pattern_id] = pattern
        self._vectors[pattern.pattern_id] = pattern.feature_vector
        if self.persist_path:
            self._save()
        return pattern.pattern_id

    def search(self, query_vector: np.ndarray, k: int = 5,
               min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar patterns.

        Args:
            query_vector: 768-dim query feature
            k: Number of results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of (pattern_id, similarity) sorted by similarity descending
        """
        if not self._vectors:
            return []

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        results = []
        for pid, vec in self._vectors.items():
            sim = float(np.dot(query_vector, vec) / (query_norm * np.linalg.norm(vec) + 1e-8))
            if sim >= min_similarity:
                results.append((pid, sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get(self, pattern_id: str) -> Optional[PatternRecord]:
        """Retrieve a pattern by ID."""
        return self._patterns.get(pattern_id)

    def delete(self, pattern_id: str) -> bool:
        """Remove a pattern from the library."""
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            del self._vectors[pattern_id]
            if self.persist_path:
                self._save()
            return True
        return False

    def list_all(self) -> List[PatternRecord]:
        """List all patterns."""
        return list(self._patterns.values())

    @property
    def size(self) -> int:
        return len(self._patterns)

    def _save(self):
        """Persist library to disk (JSON + numpy)."""
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "patterns": {},
        }
        for pid, p in self._patterns.items():
            data["patterns"][pid] = {
                "pattern_id": p.pattern_id,
                "name": p.name,
                "feature_vector": p.feature_vector.tolist(),
                "stitch_types": p.stitch_types,
                "color_count": p.color_count,
                "stitch_count": p.stitch_count,
                "created_at": p.created_at,
                "metadata": p.metadata,
            }

        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        """Load library from disk."""
        with open(self.persist_path) as f:
            data = json.load(f)

        self.dimension = data["dimension"]
        for pid, pd in data["patterns"].items():
            self._patterns[pid] = PatternRecord(
                pattern_id=pd["pattern_id"],
                name=pd["name"],
                feature_vector=np.array(pd["feature_vector"], dtype=np.float32),
                stitch_types=pd.get("stitch_types", []),
                color_count=pd.get("color_count", 0),
                stitch_count=pd.get("stitch_count", 0),
                created_at=pd.get("created_at", ""),
                metadata=pd.get("metadata", {}),
            )
            self._vectors[pid] = self._patterns[pid].feature_vector
