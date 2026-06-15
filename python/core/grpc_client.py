# ── core/grpc_client.py ──
"""
Reading-FL gRPC Client — 连接 unified-fl-backend Rust 服务

当 FedCtx 后端可用时，reading-fl 的基础设施操作委托给 Rust：
  - HNSW 向量搜索 → VectorDBService
  - FedAvg 聚合 → FederatedLearning
  - 审计链 → AuditService
  - PageRank → HybridSearchService
  - 知识图谱 → KGBuilderService / GraphRAGService
  - 文本嵌入 → EmbeddingService

当 FedCtx 不可用时，自动降级到本地 Python 实现（零配置兼容）。
"""
from __future__ import annotations

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ─── 配置 ───

FEDCTX_URL = os.environ.get("FEDCTX_URL", "http://localhost:8090")
FEDCTX_GRPC = os.environ.get("FEDCTX_GRPC", "localhost:50051")
FEDCTX_ENABLED = os.environ.get("FEDCTX_ENABLED", "auto")  # auto / on / off


@dataclass
class FedCtxConfig:
    """FedCtx 连接配置。"""
    grpc_addr: str = FEDCTX_GRPC
    rest_url: str = FEDCTX_URL
    enabled: str = FEDCTX_ENABLED  # auto = 检测后决定
    timeout: float = 5.0

    _available: Optional[bool] = field(default=None, init=False, repr=False)

    @property
    def is_available(self) -> bool:
        if self.enabled == "off":
            return False
        if self.enabled == "on":
            return True
        # auto: detect once
        if self._available is None:
            self._available = self._check_health()
        return self._available

    def _check_health(self) -> bool:
        """Check if FedCtx REST API is reachable."""
        try:
            import urllib.request
            r = urllib.request.urlopen(f"{self.rest_url}/health", timeout=2)
            return r.status == 200
        except Exception:
            return False


# ─── REST API Client (轻量，不依赖 grpcio) ───

class FedCtxRestClient:
    """
    FedCtx REST API 客户端。

    使用 REST 而非 gRPC，避免 reading-fl 需要 grpcio 编译依赖。
    unified-fl-backend 同时暴露 REST + gRPC 接口。
    """

    def __init__(self, config: FedCtxConfig = None):
        self.config = config or FedCtxConfig()

    @property
    def available(self) -> bool:
        return self.config.is_available

    # ─── Vector DB ───

    def insert_vectors(self, vectors: List[Dict]) -> int:
        """Insert vectors into HNSW index.

        Args:
            vectors: [{"id": str, "values": List[float], "metadata": dict}, ...]
        Returns:
            Number of vectors inserted.
        """
        if not self.available:
            return 0
        try:
            import urllib.request
            data = json.dumps({"vectors": vectors}).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/vectors",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout)
            resp = json.loads(r.read().decode())
            return resp.get("inserted", 0)
        except Exception as e:
            logger.warning(f"FedCtx insert_vectors failed: {e}")
            return 0

    def search_vectors(self, query: List[float], k: int = 10,
                       filter_meta: Dict[str, str] = None) -> List[Dict]:
        """Search HNSW index for nearest neighbors.

        Returns:
            [{"id": str, "distance": float, "metadata": dict}, ...]
        """
        if not self.available:
            return []
        try:
            import urllib.request
            payload = {"query": query, "k": k}
            if filter_meta:
                payload["filter"] = filter_meta
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/search",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout)
            resp = json.loads(r.read().decode())
            return resp.get("results", [])
        except Exception as e:
            logger.warning(f"FedCtx search_vectors failed: {e}")
            return []

    def text_search(self, text: str, k: int = 10) -> List[Dict]:
        """Semantic search via text (auto-embeds).

        Returns:
            [{"id": str, "distance": float, "metadata": dict}, ...]
        """
        if not self.available:
            return []
        try:
            import urllib.request
            data = json.dumps({"text": text, "k": k}).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/text_search",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout)
            resp = json.loads(r.read().decode())
            return resp.get("results", [])
        except Exception as e:
            logger.warning(f"FedCtx text_search failed: {e}")
            return []

    # ─── Federated Learning ───

    def submit_update(self, client_id: str, round_num: int,
                      parameters: np.ndarray, num_samples: int,
                      train_loss: float = 0.0) -> Dict:
        """Submit a client model update to the FL server.

        Returns:
            {"accepted": bool, "current_round": int, "updates_received": int}
        """
        if not self.available:
            return {"accepted": False, "current_round": round_num, "updates_received": 0}
        try:
            import urllib.request
            payload = {
                "client_id": client_id,
                "round": round_num,
                "parameters": parameters.flatten().tolist(),
                "num_samples": num_samples,
                "train_loss": train_loss,
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/fl/submit",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout * 2)
            return json.loads(r.read().decode())
        except Exception as e:
            logger.warning(f"FedCtx submit_update failed: {e}")
            return {"accepted": False, "current_round": round_num, "updates_received": 0}

    def aggregate(self, round_num: int, strategy: str = "fedavg",
                  min_clients: int = 1, dp_epsilon: float = 0.0) -> Optional[np.ndarray]:
        """Trigger FL aggregation and get global model.

        Returns:
            Global model parameters as numpy array, or None if failed.
        """
        if not self.available:
            return None
        try:
            import urllib.request
            payload = {
                "round": round_num,
                "strategy": strategy,
                "min_clients": min_clients,
                "dp_epsilon": dp_epsilon,
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/fl/aggregate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout * 3)
            resp = json.loads(r.read().decode())
            params = resp.get("global_parameters")
            if params:
                return np.array(params, dtype=np.float32)
            return None
        except Exception as e:
            logger.warning(f"FedCtx aggregate failed: {e}")
            return None

    # ─── Audit ───

    def append_audit(self, event_type: str, node_id: str,
                     metadata: Dict[str, str] = None) -> Optional[str]:
        """Append an entry to the audit chain.

        Returns:
            Hash of the new audit entry, or None.
        """
        if not self.available:
            return None
        try:
            import urllib.request
            payload = {
                "event_type": event_type,
                "node_id": node_id,
                "metadata": metadata or {},
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/audit/append",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout)
            resp = json.loads(r.read().decode())
            return resp.get("hash")
        except Exception as e:
            logger.warning(f"FedCtx append_audit failed: {e}")
            return None

    def get_audit_trail(self, limit: int = 50,
                        event_type: str = None) -> List[Dict]:
        """Get audit trail entries.

        Returns:
            [{"sequence": int, "timestamp": str, "event_type": str, ...}, ...]
        """
        if not self.available:
            return []
        try:
            import urllib.request
            params = f"?limit={limit}"
            if event_type:
                params += f"&event_type={event_type}"
            r = urllib.request.urlopen(
                f"{self.config.rest_url}/api/audit/trail{params}",
                timeout=self.config.timeout,
            )
            resp = json.loads(r.read().decode())
            return resp.get("entries", [])
        except Exception as e:
            logger.warning(f"FedCtx get_audit_trail failed: {e}")
            return []

    # ─── PageRank ───

    def get_pagerank_top(self, k: int = 20) -> List[Dict]:
        """Get top-k nodes by PageRank score.

        Returns:
            [{"id": str, "score": float}, ...]
        """
        if not self.available:
            return []
        try:
            import urllib.request
            r = urllib.request.urlopen(
                f"{self.config.rest_url}/api/pagerank/top?k={k}",
                timeout=self.config.timeout,
            )
            resp = json.loads(r.read().decode())
            return resp.get("results", [])
        except Exception as e:
            logger.warning(f"FedCtx get_pagerank_top failed: {e}")
            return []

    # ─── Knowledge Graph ───

    def build_kg_from_tree(self, doc_id: str, doc_name: str,
                           tree_json: str) -> Dict:
        """Build knowledge graph from PageIndex tree.

        Returns:
            {"nodes_added": int, "edges_added": int, ...}
        """
        if not self.available:
            return {"nodes_added": 0, "edges_added": 0}
        try:
            import urllib.request
            payload = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "tree_json": tree_json,
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/kg/build-tree",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout * 2)
            return json.loads(r.read().decode())
        except Exception as e:
            logger.warning(f"FedCtx build_kg_from_tree failed: {e}")
            return {"nodes_added": 0, "edges_added": 0}

    def graph_rag_query(self, query: str, top_k: int = 10,
                        max_hops: int = 2) -> Dict:
        """GraphRAG query: text → vector search → graph expansion → rerank.

        Returns:
            {"nodes": [...], "edges": [...], "total_candidates": int}
        """
        if not self.available:
            return {"nodes": [], "edges": []}
        try:
            import urllib.request
            payload = {
                "query": query,
                "top_k": top_k,
                "max_hops": max_hops,
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/graphrag/query",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout * 2)
            return json.loads(r.read().decode())
        except Exception as e:
            logger.warning(f"FedCtx graph_rag_query failed: {e}")
            return {"nodes": [], "edges": []}

    # ─── Embedding ───

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using FedCtx embedding service.

        Returns:
            List of embedding vectors.
        """
        if not self.available:
            return []
        try:
            import urllib.request
            payload = {"texts": texts}
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/embed",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout * 2)
            resp = json.loads(r.read().decode())
            return [item["values"] for item in resp.get("results", [])]
        except Exception as e:
            logger.warning(f"FedCtx embed_texts failed: {e}")
            return []

    # ─── Hybrid Search ───

    def hybrid_search(self, query: str, k: int = 10,
                      query_vector: List[float] = None) -> Dict:
        """Hybrid search: query routing → dual-channel → RRF → PageRank rerank.

        Returns:
            {"routing": {...}, "results": [...]}
        """
        if not self.available:
            return {"routing": None, "results": []}
        try:
            import urllib.request
            payload = {"query": query, "k": k}
            if query_vector:
                payload["query_vector"] = query_vector
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.config.rest_url}/api/hybrid-search",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            r = urllib.request.urlopen(req, timeout=self.config.timeout * 2)
            return json.loads(r.read().decode())
        except Exception as e:
            logger.warning(f"FedCtx hybrid_search failed: {e}")
            return {"routing": None, "results": []}

    # ─── Stats ───

    def get_stats(self) -> Dict:
        """Get FedCtx stats."""
        if not self.available:
            return {}
        try:
            import urllib.request
            r = urllib.request.urlopen(
                f"{self.config.rest_url}/api/stats",
                timeout=self.config.timeout,
            )
            return json.loads(r.read().decode())
        except Exception as e:
            logger.warning(f"FedCtx get_stats failed: {e}")
            return {}


# ─── 全局单例 ───

_default_client: Optional[FedCtxRestClient] = None


def get_fedctx_client() -> FedCtxRestClient:
    """Get or create the default FedCtx client."""
    global _default_client
    if _default_client is None:
        _default_client = FedCtxRestClient()
    return _default_client
