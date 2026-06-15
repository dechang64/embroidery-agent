"""
Embroidery Agent — 刺绣针迹自动生成系统 MVP v2

整合 embodied-fl 技术栈：
    - DINOv2 特征提取 → 刺绣图案风格指纹（复用 embodied-fl/feature_extractor.py）
    - HNSW 向量索引 → 刺绣图案库检索（复用 embodied-fl/hnsw_index.rs）
    - YOLOv11 检测 → 刺绣区域智能分割（复用 embodied-fl/detector.py）
    - 区块链审计链 → 刺绣设计版权存证（复用 embodied-fl/audit.rs）
    - gRPC 通信 → 多工坊协同（复用 embodied-fl/grpc_service.rs）

Pipeline:
    Image → DINOv2 Style Fingerprint → HNSW Pattern Search →
    YOLOv11 Region Detection → Stitch Planning → PES/DST Export →
    Audit Chain Certification
"""

from .agent import EmbroideryAgent, GenerationResult
from .image_processor import ImageProcessor, ProcessedImage, ImageRegion, StitchType, EmbroideryColor
from .stitch_planner import StitchPlanner, StitchPlan, StitchBlock, StitchPoint
from .pattern_generator import PatternGenerator, ExportResult
from .style_fingerprint import StyleFingerprint, PatternLibrary
from .audit_certifier import AuditCertifier, DesignCertificate

__version__ = "0.2.0"
__all__ = [
    "EmbroideryAgent", "GenerationResult",
    "ImageProcessor", "ProcessedImage", "ImageRegion", "StitchType", "EmbroideryColor",
    "StitchPlanner", "StitchPlan", "StitchBlock", "StitchPoint",
    "PatternGenerator", "ExportResult",
    "StyleFingerprint", "PatternLibrary",
    "AuditCertifier", "DesignCertificate",
]
