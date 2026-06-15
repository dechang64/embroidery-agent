"""
FedAvg Aggregator — federated averaging for embroidery stitch models.

Mirrors embodied-fl/fed_server.rs aggregation logic.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ClientUpdate:
    """A client's model update for aggregation."""
    client_id: str
    round_id: int
    num_samples: int
    local_loss: float
    weights: List[List[float]]
    stitch_type: str = "satin"


@dataclass
class AggregationResult:
    """Result of FedAvg aggregation."""
    global_weights: List[List[float]]
    global_loss: float
    participating_clients: int
    total_samples: int


class FedAvgAggregator:
    """Federated Averaging aggregator.

    Algorithm:
        1. Collect weight updates from all clients
        2. Compute weighted average: w_global = Σ(n_k / N) * w_k
        3. Return aggregated weights
    """

    def aggregate(self, updates: List[ClientUpdate]) -> AggregationResult:
        """Perform FedAvg aggregation over client updates.

        FedCtx integration: when unified-fl-backend is available, delegates
        aggregation to the Rust server (supports FedAvg/FedProx/EWA + DP).
        Falls back to local Python FedAvg when FedCtx is unavailable.
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        # Try FedCtx aggregation
        try:
            from core.grpc_client import get_fedctx_client
            client = get_fedctx_client()
            if client.available:
                for i, u in enumerate(updates):
                    flat = []
                    for layer in u.weights:
                        flat.extend(layer)
                    client.fl_submit_update(
                        client_id=f"embroidery_client_{u.client_id}",
                        round_num=u.round_id,
                        parameters=flat,
                        num_samples=u.num_samples,
                    )
                agg_resp = client.fl_aggregate(strategy="fedavg")
                if agg_resp and agg_resp.get("parameters"):
                    agg_flat = np.array(agg_resp["parameters"])
                    offset = 0
                    global_weights = []
                    for layer in updates[0].weights:
                        size = len(layer)
                        global_weights.append(agg_flat[offset:offset + size].tolist())
                        offset += size
                    total_samples = sum(u.num_samples for u in updates)
                    global_loss = sum(u.local_loss * u.num_samples for u in updates) / max(total_samples, 1)
                    return AggregationResult(
                        global_weights=global_weights,
                        global_loss=global_loss,
                        participating_clients=len(updates),
                        total_samples=total_samples,
                    )
        except (ImportError, Exception):
            pass  # Fall through to local aggregation

        # Local fallback: Python FedAvg
        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            raise ValueError("Total samples is zero — cannot aggregate")
        global_loss = 0.0

        # Initialize with zeros
        weight_dim = len(updates[0].weights)
        agg_weights = [np.zeros_like(np.array(w)) for w in updates[0].weights]

        for update in updates:
            weight = update.num_samples / total_samples
            global_loss += update.local_loss * weight
            for i, w in enumerate(update.weights):
                agg_weights[i] += np.array(w) * weight

        return AggregationResult(
            global_weights=[w.tolist() for w in agg_weights],
            global_loss=global_loss,
            participating_clients=len(updates),
            total_samples=total_samples,
        )

    def compute_convergence(self, history: List[AggregationResult]) -> Dict[str, Any]:
        """Compute convergence metrics from aggregation history."""
        if len(history) < 2:
            return {"converged": False, "reason": "insufficient_history"}

        losses = [h.global_loss for h in history]
        recent = losses[-5:] if len(losses) >= 5 else losses

        # Check if loss is decreasing
        decreasing = all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1))

        # Check if loss plateaued
        loss_range = max(recent) - min(recent)
        loss_mean = np.mean(recent)
        plateaued = loss_range < 0.01 * abs(loss_mean) if loss_mean != 0 else loss_range < 0.001

        return {
            "converged": decreasing and plateaued,
            "loss_trend": "decreasing" if decreasing else "increasing",
            "plateaued": plateaued,
            "latest_loss": losses[-1],
            "loss_improvement": losses[0] - losses[-1],
        }
