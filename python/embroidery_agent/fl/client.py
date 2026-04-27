"""
Federated Learning Client — multi-workshop embroidery stitch optimization.

Each workshop (embroidery factory) trains locally on its proprietary designs,
then submits weight updates to the central server. The server aggregates using
FedAvg to produce a global stitch quality model — no raw design data leaves
the workshop.

Mirrors embodied-fl/fed_server.rs client interface.
"""

import numpy as np
import requests
import json
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkshopConfig:
    """Configuration for a workshop (embroidery factory)."""
    workshop_id: str
    workshop_name: str
    specialty: str  # e.g., "satin", "fill", "chain", "tatami"
    num_samples: int = 500
    model_dim: int = 128
    local_epochs: int = 5
    learning_rate: float = 0.01
    api_base: str = "http://localhost:8080/api/v1"


class StitchQualityModel:
    """Simple MLP for predicting stitch quality (pure NumPy, no PyTorch dependency)."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 9):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.forward(x)
        return float(np.mean((pred - y) ** 2))

    def compute_gradients(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        m = x.shape[0]
        pred = self.forward(x)
        dz2 = 2 * (pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return [dW1, db1, dW2, db2]

    def apply_gradients(self, grads: List[np.ndarray], lr: float = 0.01):
        for param, grad in zip(self.params, grads):
            param -= lr * grad

    def get_weights(self) -> List[List[float]]:
        return [w.tolist() for w in self.params]

    def set_weights(self, weights: List[List[float]]):
        for param, w in zip(self.params, weights):
            param[:] = np.array(w)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.forward(x)
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        return float(np.mean(pred_labels == true_labels))


class FederatedClient:
    """Federated learning client for a workshop.

    Workflow:
        1. register() — register with server
        2. download_global_model() — get current global weights
        3. train_local() — train on local data
        4. submit_update() — send weights to server
        5. Repeat for each round
    """

    def __init__(self, config: WorkshopConfig):
        self.config = config
        self.model = StitchQualityModel(
            input_dim=config.model_dim,
            output_dim=9,  # 9 stitch types
        )
        self.prev_loss = float("inf")

    def register(self) -> bool:
        """Register this workshop with the server."""
        try:
            resp = requests.post(f"{self.config.api_base}/fed/register", json={
                "client_id": self.config.workshop_id,
                "workshop_name": self.config.workshop_name,
                "specialty": self.config.specialty,
            })
            print(f"  [{self.config.workshop_id}] Registered: {resp.json()}")
            return resp.status_code == 200
        except Exception as e:
            print(f"  [{self.config.workshop_id}] Registration failed: {e}")
            return False

    def download_global_model(self, round_id: int = 0) -> bool:
        """Download global model weights from server."""
        try:
            resp = requests.get(f"{self.config.api_base}/fed/model", params={"client_id": self.config.workshop_id})
            data = resp.json()
            if data.get("weights"):
                self.model.set_weights(data["weights"])
                print(f"  [{self.config.workshop_id}] Downloaded global model (round {data.get('round_id', '?')})")
                return True
        except Exception as e:
            print(f"  [{self.config.workshop_id}] Download failed: {e}")
        return False

    def train_local(self, epochs: Optional[int] = None, lr: Optional[float] = None) -> float:
        """Train locally on synthetic workshop data."""
        epochs = epochs or self.config.local_epochs
        lr = lr or self.config.learning_rate

        # Generate synthetic training data (in production, use real workshop designs)
        rng = np.random.RandomState(hash(self.config.workshop_id) % 2 ** 31)
        X = rng.randn(self.config.num_samples, self.config.model_dim).astype(np.float32)
        y = np.zeros((self.config.num_samples, 9), dtype=np.float32)
        labels = rng.randint(0, 9, self.config.num_samples)
        y[np.arange(self.config.num_samples), labels] = 1.0

        for epoch in range(epochs):
            idx = rng.choice(self.config.num_samples, min(32, self.config.num_samples), replace=False)
            grads = self.model.compute_gradients(X[idx], y[idx])
            self.model.apply_gradients(grads, lr)

        loss = self.model.compute_loss(X, y)
        return loss

    def submit_update(self, round_id: int, local_loss: float) -> bool:
        """Submit local model update to server."""
        try:
            resp = requests.post(f"{self.config.api_base}/fed/update", json={
                "client_id": self.config.workshop_id,
                "round_id": round_id,
                "num_samples": self.config.num_samples,
                "local_loss": local_loss,
                "weights": self.model.get_weights(),
                "stitch_type": self.config.specialty,
            })
            accepted = resp.json().get("accepted", False)
            improvement = self.prev_loss - local_loss
            print(f"  [{self.config.workshop_id}] Update: loss={local_loss:.4f}, Δ={improvement:.4f}, accepted={accepted}")
            self.prev_loss = local_loss
            return accepted
        except Exception as e:
            print(f"  [{self.config.workshop_id}] Submit failed: {e}")
            return False

    def run_fed_round(self, round_id: int) -> Dict[str, Any]:
        """Execute one full federated round."""
        self.download_global_model(round_id)
        loss = self.train_local()
        accepted = self.submit_update(round_id, loss)
        return {"workshop_id": self.config.workshop_id, "round_id": round_id,
                "loss": loss, "accepted": accepted}
