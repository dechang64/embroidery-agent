"""Federated learning module for multi-workshop embroidery optimization."""

from .client import FederatedClient, WorkshopConfig
from .aggregation import FedAvgAggregator

__all__ = ["FederatedClient", "WorkshopConfig", "FedAvgAggregator"]
