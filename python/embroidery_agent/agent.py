"""
Main Embroidery Agent — orchestrates the full pipeline.

Pipeline: Image → DINOv2 Fingerprint → Pattern Search → Process →
          Plan Stitches → Export Files → Audit Certify

Usage:
    from embroidery_agent import EmbroideryAgent

    agent = EmbroideryAgent()
    result = agent.generate("input.png", output_dir="./output")
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image

from .image_processor import ImageProcessor
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
    regions_count: int = 0
    processing_time_ms: float = 0.0
    certificate: Optional[DesignCertificate] = None
    similar_patterns: List[Tuple[str, float]] = field(default_factory=list)
    style_hash: str = ""

    @property
    def summary(self) -> str:
        lines = [
            f"Embroidery Generation Complete",
            f"  Regions: {self.regions_count}",
            f"  Stitches: {self.stitch_plan.total_stitches:,}",
            f"  Colors: {self.stitch_plan.total_colors}",
            f"  Size: {self.stitch_plan.design_width_mm:.1f} × {self.stitch_plan.design_height_mm:.1f} mm",
            f"  Time: {self.processing_time_ms:.0f} ms",
        ]
        if self.style_hash:
            lines.append(f"  Style Hash: {self.style_hash}")
        if self.similar_patterns:
            lines.append(f"  Similar Patterns: {len(self.similar_patterns)}")
            for pid, sim in self.similar_patterns[:3]:
                lines.append(f"    {pid}: {sim:.3f}")
        if self.certificate:
            lines.append(f"  Certificate: {self.certificate.audit_hash[:16]}...")
        lines.append(f"  Files:")
        for exp in self.exports:
            lines.append(f"    {exp.format}: {exp.file_path} ({exp.file_size_bytes:,} bytes)")
        return "\n".join(lines)


class EmbroideryAgent:
    """Main agent for embroidery pattern generation.

    Orchestrates the full pipeline:
        Image → DINOv2 Fingerprint → Pattern Library Search →
        Image Processing → Stitch Planning → File Export → Audit Certification

    Integrates embodied-fl tech stack:
        - StyleFingerprint (from embodied-fl DINOv2SceneExtractor)
        - PatternLibrary (from embodied-fl HNSW index)
        - AuditCertifier (from embodied-fl AuditChain)
    """

    def __init__(self, max_colors: int = 8, stitch_density: float = 3.0,
                 resolution: float = 5.0, export_formats: List[str] = None,
                 enable_fingerprint: bool = True, enable_audit: bool = True,
                 pattern_library_path: Optional[str] = None,
                 audit_db_path: Optional[str] = None):
        """
        Args:
            max_colors: Maximum number of colors to extract
            stitch_density: Stitches per mm for fill patterns
            resolution: Pixels per mm for coordinate conversion
            export_formats: List of formats to export (pes, dst, svg, etc.)
            enable_fingerprint: Enable DINOv2 style fingerprinting
            enable_audit: Enable blockchain audit certification
            pattern_library_path: Path for pattern library persistence
            audit_db_path: Path for audit database
        """
        self.processor = ImageProcessor(max_colors=max_colors)
        self.planner = StitchPlanner(
            stitch_density=stitch_density,
            resolution=resolution,
        )
        self.generator = PatternGenerator()
        self.export_formats = export_formats or ["pes", "dst", "svg"]

        # embodied-fl integrations
        self.enable_fingerprint = enable_fingerprint
        self.enable_audit = enable_audit

        if enable_fingerprint:
            self.fingerprint = StyleFingerprint()
            self.pattern_library = PatternLibrary(
                persist_path=pattern_library_path,
            )
        else:
            self.fingerprint = None
            self.pattern_library = None

        if enable_audit:
            self.audit = AuditCertifier(db_path=audit_db_path or "embroidery_audit.db")
        else:
            self.audit = None

    def generate(self, input_path: str, output_dir: str = "./output",
                 name: str = "design", designer_id: str = "anonymous",
                 save_to_library: bool = False) -> GenerationResult:
        """Full pipeline: image → embroidery files + audit certificate.

        Args:
            input_path: Path to input image (PNG, JPG, etc.)
            output_dir: Directory for output files
            name: Base name for output files
            designer_id: Designer identifier for audit chain
            save_to_library: Whether to save pattern to library

        Returns:
            GenerationResult with all outputs
        """
        import time
        start = time.time()

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base_path = str(Path(output_dir) / name)

        # Step 0: DINOv2 style fingerprint (embodied-fl integration)
        style_hash = ""
        similar_patterns = []
        if self.fingerprint:
            feature = self.fingerprint.extract(input_path)
            style_hash = self.fingerprint.compute_hash(feature)[:16]
            similar_patterns = self.pattern_library.search(feature, k=5)

        # Step 1: Load and process image
        image = Image.open(input_path).convert("RGB")
        processed = self.processor.process(image)

        # Step 2: Plan stitches
        plan = self.planner.plan(processed.regions, processed.color_palette)

        # Step 3: Export files
        exports = self.generator.export_multi_format(plan, base_path, self.export_formats)

        # Step 4: Generate preview
        preview_path = f"{base_path}_preview.svg"
        self.generator.generate_preview_svg(plan, preview_path)

        # Step 5: Audit certification (embodied-fl integration)
        certificate = None
        if self.audit:
            certificate = self.audit.certify_design(
                design_id=name,
                designer_id=designer_id,
                stitch_count=plan.total_stitches,
                color_count=plan.total_colors,
                file_formats=[e.format for e in exports],
                design_hash=style_hash,
            )

        # Step 6: Save to pattern library
        if save_to_library and self.fingerprint:
            feature = self.fingerprint.extract(input_path)
            record = PatternRecord(
                pattern_id=name,
                name=name,
                feature_vector=feature,
                stitch_types=[b.stitch_type.value for b in plan.blocks],
                color_count=plan.total_colors,
                stitch_count=plan.total_stitches,
                created_at=certificate.created_at if certificate else "",
                metadata={"designer": designer_id, "exports": len(exports)},
            )
            self.pattern_library.add(record)

        elapsed_ms = (time.time() - start) * 1000

        return GenerationResult(
            input_image=input_path,
            output_dir=output_dir,
            stitch_plan=plan,
            exports=exports,
            preview_svg=preview_path,
            regions_count=len(processed.regions),
            processing_time_ms=elapsed_ms,
            certificate=certificate,
            similar_patterns=similar_patterns,
            style_hash=style_hash,
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
        preview_path = f"{base_path}_preview.svg"
        self.generator.generate_preview_svg(plan, preview_path)

        elapsed_ms = (time.time() - start) * 1000

        return GenerationResult(
            input_image="<array>",
            output_dir=output_dir,
            stitch_plan=plan,
            exports=exports,
            preview_svg=preview_path,
            regions_count=len(processed.regions),
            processing_time_ms=elapsed_ms,
        )
