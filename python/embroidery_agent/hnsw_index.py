"""
Embroidery Agent HNSW Index — Pure NumPy implementation

Lightweight HNSW (Hierarchical Navigable Small World) implementation
for embroidery pattern style matching and recommendation.

Ported from reading-fl/matching/hnsw_index.py — our own pure-NumPy HNSW,
no external dependencies (no hnsw crate, no space crate).

Pure NumPy, no external dependencies.
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class HNSWIndex:
    """
    HNSW vector index for approximate nearest neighbor search.

    Used for:
    1. Pattern matching: find patterns with similar style fingerprints
    2. Design recommendation: find designs similar to a reference
    3. Workshop clustering: group similar workshop specialties for FL

    Parameters:
        dim: Vector dimensionality
        M: Max connections per node per layer
        ef_construction: Beam width during construction
        ef_search: Beam width during search
    """

    def __init__(
        self,
        dim: int = 128,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_elements: int = 10000,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M  # Max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements

        # Storage
        self.vectors: Dict[int, np.ndarray] = {}
        self.metadata: Dict[int, dict] = {}

        # Graph: layer -> node -> set of neighbor node ids
        self.graph: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))

        # Node -> max layer
        self.node_layers: Dict[int, int] = {}

        # Entry point
        self.entry_point: Optional[int] = None
        self.max_layer: int = -1

        self._n_elements = 0

    def _random_level(self) -> int:
        """Generate random level using exponential distribution."""
        level = 0
        while np.random.random() < 1.0 / (self.M + 1) and level < 16:
            level += 1
        return level

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance."""
        return float(np.linalg.norm(a - b))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """
        Search a single layer, returning ef nearest neighbors.

        Returns:
            List of (distance, node_id) tuples, sorted by distance ascending.
        """
        visited = set(entry_points)
        candidates = []  # Min-heap: (distance, id)
        results = []     # Max-heap: (-distance, id)

        for ep in entry_points:
            if ep not in self.vectors:
                continue
            dist = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            # If closest candidate is farther than farthest result, stop
            if dist_c > -results[0][0]:
                break

            # Explore neighbors
            neighbors = self.graph[layer].get(c, set())
            for n in neighbors:
                if n in visited:
                    continue
                visited.add(n)

                if n not in self.vectors:
                    continue

                dist_n = self._distance(query, self.vectors[n])
                farthest = -results[0][0]

                if dist_n < farthest or len(results) < ef:
                    heapq.heappush(candidates, (dist_n, n))
                    heapq.heappush(results, (-dist_n, n))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(abs(d), idx) for d, idx in sorted(results)]

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, int]],
        M: int,
    ) -> List[Tuple[float, int]]:
        """Select M nearest neighbors from candidates (simple selection)."""
        return sorted(candidates)[:M]

    def add(self, vector: np.ndarray, idx: Optional[int] = None, metadata: dict = None) -> int:
        """
        Add a vector to the index.

        Args:
            vector: The vector to add (dim,)
            idx: Optional custom ID. If None, auto-incremented.
            metadata: Optional metadata dict (e.g., reader_id, campus_id)

        Returns:
            The assigned index.
        """
        if idx is None:
            idx = self._n_elements

        if self._n_elements >= self.max_elements:
            raise ValueError(f"Index full (max_elements={self.max_elements})")

        self.vectors[idx] = vector.astype(np.float32)
        self.metadata[idx] = metadata or {}
        self._n_elements += 1

        # Assign random layer
        level = self._random_level()
        self.node_layers[idx] = level

        # Initialize graph entries
        for l in range(level + 1):
            self.graph[l][idx] = set()

        if self.entry_point is None:
            # First element
            self.entry_point = idx
            self.max_layer = level
            return idx

        # Find entry point for insertion
        ep = [self.entry_point]

        # Traverse from top to insertion layer + 1
        for l in range(self.max_layer, level, -1):
            results = self._search_layer(vector, ep, ef=1, layer=l)
            ep = [results[0][1]]

        # Insert at layers [level, 0]
        for l in range(min(level, self.max_layer), -1, -1):
            M_max = self.M_max0 if l == 0 else self.M
            results = self._search_layer(vector, ep, ef=self.ef_construction, layer=l)
            neighbors = self._select_neighbors(vector, results, M_max)

            # Add bidirectional connections
            for dist, n_idx in neighbors:
                self.graph[l][idx].add(n_idx)
                self.graph[l][n_idx].add(idx)

                # Prune if too many connections
                if len(self.graph[l][n_idx]) > M_max:
                    # Keep M_max nearest
                    n_vec = self.vectors[n_idx]
                    n_neighbors = [
                        (self._distance(n_vec, self.vectors[nn]), nn)
                        for nn in self.graph[l][n_idx]
                        if nn in self.vectors
                    ]
                    n_neighbors = self._select_neighbors(n_vec, n_neighbors, M_max)
                    self.graph[l][n_idx] = set(nn for _, nn in n_neighbors)

            ep = [idx for _, idx in results]

        # Update entry point if new node has higher layer
        if level > self.max_layer:
            self.entry_point = idx
            self.max_layer = level

        return idx

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, int, dict]]:
        """
        Search for k nearest neighbors.

        Returns:
            List of (distance, index, metadata) tuples.
        """
        if self.entry_point is None:
            return []

        ep = [self.entry_point]

        # Traverse from top layer to layer 1
        for l in range(self.max_layer, 0, -1):
            results = self._search_layer(query, ep, ef=1, layer=l)
            ep = [results[0][1]]

        # Search at layer 0 with full ef
        results = self._search_layer(query, ep, ef=self.ef_search, layer=0)

        return [
            (dist, idx, self.metadata.get(idx, {}))
            for dist, idx in results[:k]
        ]

    def search_by_metadata(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_key: str = None,
        filter_value = None,
    ) -> List[Tuple[float, int, dict]]:
        """Search with metadata filtering."""
        results = self.search(query, k=k * 3)  # Over-fetch
        if filter_key is None:
            return results[:k]

        filtered = [
            r for r in results
            if self.metadata.get(r[1], {}).get(filter_key) == filter_value
        ]
        return filtered[:k]

    def remove(self, idx: int):
        """Remove a vector from the index."""
        if idx not in self.vectors:
            return
        del self.vectors[idx]
        self.metadata.pop(idx, None)
        self.node_layers.pop(idx, None)
        for layer_graph in self.graph.values():
            layer_graph.pop(idx, None)
            for neighbors in layer_graph.values():
                neighbors.discard(idx)

    def __len__(self) -> int:
        return self._n_elements

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self.dim}, elements={self._n_elements}, "
            f"layers={self.max_layer + 1}, M={self.M})"
        )
