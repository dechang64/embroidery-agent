"""
Main Embroidery Agent — orchestrates the full pipeline.

Pipeline: Image → Quantize → Region Extract → Plan Stitches → Export → Preview → Certify
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from PIL import Image

from .image_processor import ImageProcessor, ProcessedImage
from .stitch_planner import StitchPlanner, StitchPlan
from .pattern_generator import PatternGenerator, ExportResult
from .style_fingerprint import StyleFingerprint, PatternLibrary, PatternRecord
from .audit_certifier import AuditCertifier, DesignCertificate


@dataclass
class GenerationResult:
    """Complete result of embroidery generation."""
    input_image: str
    output_dir: str
    stitch_plan: StitchPlan
    exports: List[ExportResult] = field(default_factory=list)
    preview_svg: str = ""
    preview_png: str = ""
    regions_count: int = 0
    processing_time_ms: float = 0.0
    certificate: Optional[DesignCertificate] = None
    style_hash: str = ""
    similar_patterns: List[Dict[str, Any]] = field(default_factory=list)


class EmbroideryAgent:
    """Embroidery stitch auto-generation agent.

    Usage:
        from embroidery_agent import EmbroideryAgent
        agent = EmbroideryAgent()
        result = agent.generate("input.png", output_dir="./output")
    """

    def __init__(self, api_base: str = "http://localhost:8080/api/v1",
                 audit_db: str = "embroidery_audit.db",
                 pattern_db: str = "pattern_library.json"):
        self.api_base = api_base
        self.processor = ImageProcessor()
        self.planner = StitchPlanner()
        self.generator = PatternGenerator()
        self.fingerprint = StyleFingerprint()
        self.pattern_library = PatternLibrary(persist_path=pattern_db)
        self.audit = AuditCertifier(db_path=audit_db)
        self.export_formats = ["pes", "dst", "svg"]

    def generate(self, image_path: str, output_dir: str = "./output",
                 name: Optional[str] = None, certify: bool = True) -> GenerationResult:
        """Full pipeline: image → stitches → export → preview → certify."""
        import time
        start = time.time()

        path = Path(image_path)
        name = name or path.stem
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_path = str(Path(output_dir) / name)

        # 1. Load and process image
        image = Image.open(image_path).convert("RGB")
        processed = self.processor.process(image)

        # 2. Style fingerprinting
        try:
            feature = self.fingerprint.compute(image)
            style_hash = StyleFingerprint.style_hash(feature)
            similar = self.pattern_library.search(feature, top_k=3)
            similar_patterns = [{"pattern_id": s[0], "similarity": float(s[1])} for s in similar]
        except Exception:
            style_hash = ""
            similar_patterns = []

        # 3. Plan stitches
        plan = self.planner.plan(processed.regions, processed.color_palette)

        # 4. Export machine files
        exports = self.generator.export_multi_format(plan, base_path, self.export_formats)

        # 5. Generate previews
        preview_svg_path = f"{base_path}_preview.svg"
        self.generator.generate_preview_svg(plan, preview_svg_path)

        preview_png_path = f"{base_path}_preview.png"
        self.generator.generate_preview_png(plan, preview_png_path,
                                           regions=processed.regions,
                                           processed=processed)

        # 6. Audit certification
        certificate = None
        if certify:
            design_hash = style_hash or self._hash_file(image_path)
            certificate = self.audit.certify_design(
                design_hash=design_hash,
                designer_id="agent",
                stitch_count=plan.total_stitches,
                color_count=plan.total_colors,
            )

        elapsed_ms = (time.time() - start) * 1000

        return GenerationResult(
            input_image=image_path, output_dir=output_dir,
            stitch_plan=plan, exports=exports,
            preview_svg=preview_svg_path, preview_png=preview_png_path,
            regions_count=len(processed.regions), processing_time_ms=elapsed_ms,
            certificate=certificate, style_hash=style_hash,
            similar_patterns=similar_patterns,
        )

    def generate_from_array(self, image_array, output_dir: str = "./output",
                            name: str = "design") -> GenerationResult:
        """Generate from numpy array instead of file path."""
        import time
        start = time.time()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_path = str(Path(output_dir) / name)

        image = Image.fromarray(image_array).convert("RGB")
        processed = self.processor.process(image)
        plan = self.planner.plan(processed.regions, processed.color_palette)
        exports = self.generator.export_multi_format(plan, base_path, self.export_formats)

        preview_svg_path = f"{base_path}_preview.svg"
        self.generator.generate_preview_svg(plan, preview_svg_path)
        preview_png_path = f"{base_path}_preview.png"
        self.generator.generate_preview_png(plan, preview_png_path,
                                           regions=processed.regions,
                                           processed=processed)

        elapsed_ms = (time.time() - start) * 1000
        return GenerationResult(input_image="<array>", output_dir=output_dir,
                                stitch_plan=plan, exports=exports,
                                preview_svg=preview_svg_path, preview_png=preview_png_path,
                                regions_count=len(processed.regions), processing_time_ms=elapsed_ms)

    @staticmethod
    def _hash_file(path: str) -> str:
        import hashlib
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
