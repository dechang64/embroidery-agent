"""
Pattern generator module for Embroidery Agent.

Generates standard embroidery machine file formats:
    - PES (Brother/Baby Lock)
    - DST (Tajima)
    - EXP (Melco)
    - SVG (preview)

Uses pyembroidery for format encoding.
"""

import pyembroidery
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from .image_processor import EmbroideryColor, StitchType
from .stitch_planner import StitchPlan, StitchBlock, StitchPoint


@dataclass
class ExportResult:
    """Result of pattern export."""
    format: str
    file_path: str
    stitch_count: int
    color_count: int
    design_width_mm: float
    design_height_mm: float
    file_size_bytes: int = 0


class PatternGenerator:
    """Generates embroidery machine files from stitch plans.

    Supports PES, DST, EXP, JEF, SVG formats.
    """

    def __init__(self, dpi: float = 254.0):
        """
        Args:
            dpi: Dots per inch for coordinate conversion.
                 254 DPI = 10 pixels/mm (standard embroidery resolution).
        """
        self.dpi = dpi

    def export(self, plan: StitchPlan, output_path: str,
               format: str = "pes") -> ExportResult:
        """Export stitch plan to embroidery file format.

        Args:
            plan: Complete stitch plan
            output_path: Output file path (without extension)
            format: File format (pes, dst, exp, jef, svg)

        Returns:
            ExportResult with metadata
        """
        # Build pyembroidery pattern
        pattern = pyembroidery.EmbPattern()

        # Add stitches block by block
        for block_idx, block in enumerate(plan.blocks):
            # Color change (except first block)
            if block_idx > 0:
                pattern.color_change()

            # Set thread color
            if block.color:
                thread = pyembroidery.EmbThread()
                thread.set_color(block.color.rgb[0], block.color.rgb[1], block.color.rgb[2])
                thread.description = block.color.name
                thread.catalog_number = block.color.thread_code
                pattern.add_thread(thread)

            # Add stitch points
            for point in block.points:
                # Convert mm to 0.1mm units (pyembroidery default)
                x_01mm = point.x * 10
                y_01mm = point.y * 10

                if point.jump:
                    pattern.trim()
                    pattern.move_abs(x_01mm, y_01mm)
                else:
                    pattern.stitch_abs(x_01mm, y_01mm)

        # End of design
        pattern.end()

        # Write file
        ext = format.lower().lstrip('.')
        full_path = f"{output_path}.{ext}"

        pyembroidery.write(pattern, full_path)

        # Get file size
        file_size = Path(full_path).stat().st_size if Path(full_path).exists() else 0

        return ExportResult(
            format=ext.upper(),
            file_path=full_path,
            stitch_count=plan.total_stitches,
            color_count=plan.total_colors,
            design_width_mm=plan.design_width_mm,
            design_height_mm=plan.design_height_mm,
            file_size_bytes=file_size,
        )

    def export_multi_format(self, plan: StitchPlan, output_path: str,
                             formats: List[str] = None) -> List[ExportResult]:
        """Export to multiple formats at once."""
        if formats is None:
            formats = ["pes", "dst", "svg"]

        results = []
        for fmt in formats:
            result = self.export(plan, output_path, fmt)
            results.append(result)
        return results

    def generate_preview_svg(self, plan: StitchPlan, output_path: str,
                              width: int = 500, height: int = 500) -> str:
        """Generate an SVG preview of the stitch plan."""
        import svgwrite

        if not plan.blocks:
            return ""

        dwg = svgwrite.Drawing(output_path, size=(f"{width}px", f"{height}px"))

        # Calculate scale
        if plan.design_width_mm > 0 and plan.design_height_mm > 0:
            scale_x = (width - 20) / plan.design_width_mm
            scale_y = (height - 20) / plan.design_height_mm
            scale = min(scale_x, scale_y)
        else:
            scale = 1.0

        # Offset to center
        offset_x = 10
        offset_y = 10

        # Draw stitch blocks
        for block in plan.blocks:
            if not block.points:
                continue

            color_hex = block.color.hex if block.color else "#000000"

            # Draw stitch lines
            for i in range(1, len(block.points)):
                p1 = block.points[i - 1]
                p2 = block.points[i]

                if p1.jump or p2.jump:
                    continue

                x1 = offset_x + p1.x * scale
                y1 = offset_y + p1.y * scale
                x2 = offset_x + p2.x * scale
                y2 = offset_y + p2.y * scale

                dwg.add(dwg.line(
                    start=(x1, y1), end=(x2, y2),
                    stroke=color_hex,
                    stroke_width=0.5,
                    opacity=0.8,
                ))

        dwg.save()
        return output_path
