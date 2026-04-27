"""
FedAvg Aggregator — federated averaging for embroidery stitch models.

Mirrors embodied-fl/fed_server.rs aggregation logic.
"""

import numpy as np
from typing import List, Dict, Any, Optional
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
        """Perform FedAvg aggregation over client updates."""
        if not updates:
            raise ValueError("No updates to aggregate")

        total_samples = sum(u.num_samples for u in updates)
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
