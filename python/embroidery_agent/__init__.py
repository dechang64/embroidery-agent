"""
Embroidery Agent — Python client package.

Connects to Rust backend via gRPC/REST for:
  - Style fingerprinting (DINOv2 768-dim)
  - Pattern library search (HNSW)
  - Stitch planning and export (PES/DST/EXP)
  - Audit chain certification
  - Federated learning (multi-workshop)
"""

from .image_processor import ImageProcessor, ProcessedImage, StitchType, EmbroideryColor, ImageRegion
from .stitch_planner import StitchPlanner, StitchPlan, StitchBlock, StitchPoint
from .pattern_generator import PatternGenerator, ExportResult
from .style_fingerprint import StyleFingerprint, PatternLibrary, PatternRecord
from .hnsw_index import HNSWIndex
from .audit_certifier import AuditCertifier, DesignCertificate, AuditEntry
from .fl.client import FederatedClient, WorkshopConfig
from .fl.aggregation import FedAvgAggregator

__version__ = "0.2.0"
__all__ = [
    "ImageProcessor", "ProcessedImage", "StitchType", "EmbroideryColor", "ImageRegion",
    "StitchPlanner", "StitchPlan", "StitchBlock", "StitchPoint",
    "PatternGenerator", "ExportResult",
    "StyleFingerprint", "PatternLibrary", "PatternRecord",
    "HNSWIndex",
    "AuditCertifier", "DesignCertificate", "AuditEntry",
    "FederatedClient", "WorkshopConfig", "FedAvgAggregator",
]
