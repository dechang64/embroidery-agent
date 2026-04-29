"""
Pattern generator module for Embroidery Agent.

Generates standard embroidery machine file formats:
    - PES (Brother/Baby Lock)
    - DST (Tajima)
    - EXP (Melco)
    - SVG (preview)
"""

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
    """Generates embroidery machine files from stitch plans."""

    def __init__(self, dpi: float = 254.0):
        self.dpi = dpi  # 254 DPI = 10 pixels/mm

    def export_multi_format(self, plan: StitchPlan, base_path: str,
                            formats: Optional[List[str]] = None) -> List[ExportResult]:
        """Export stitch plan to multiple formats."""
        if formats is None:
            formats = ["pes", "dst", "svg"]
        results = []
        for fmt in formats:
            try:
                result = self.export(plan, base_path, fmt)
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  Export to {fmt} failed: {e}")
        return results

    def export(self, plan: StitchPlan, base_path: str, fmt: str) -> ExportResult:
        """Export stitch plan to a single format."""
        path = f"{base_path}.{fmt}"
        if fmt == "svg":
            self._export_svg(plan, path)
        elif fmt == "dst":
            self._export_dst(plan, path)
        elif fmt == "pes":
            self._export_pes(plan, path)
        elif fmt == "exp":
            self._export_exp(plan, path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        file_size = Path(path).stat().st_size if Path(path).exists() else 0
        return ExportResult(
            format=fmt.upper(),
            file_path=path,
            stitch_count=plan.total_stitches,
            color_count=plan.total_colors,
            design_width_mm=plan.design_width_mm,
            design_height_mm=plan.design_height_mm,
            file_size_bytes=file_size,
        )

    def _export_svg(self, plan: StitchPlan, output_path: str):
        """Generate SVG preview of stitch plan."""
        width = max(200, int(plan.design_width_mm * 10) + 20)
        height = max(200, int(plan.design_height_mm * 10) + 20)
        scale = min((width - 20) / max(plan.design_width_mm * 10, 1),
                    (height - 20) / max(plan.design_height_mm * 10, 1))

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="100%" height="100%" fill="white"/>',
        ]

        for block in plan.blocks:
            if not block.points:
                continue
            color_hex = block.color.hex if block.color else "#000000"
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                if p1.jump or p2.jump:
                    continue
                x1, y1 = 10 + p1.x * scale, 10 + p1.y * scale
                x2, y2 = 10 + p2.x * scale, 10 + p2.y * scale
                lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                             f'stroke="{color_hex}" stroke-width="0.5" opacity="0.8"/>')

        lines.append("</svg>")
        Path(output_path).write_text("\n".join(lines))

    def _export_dst(self, plan: StitchPlan, output_path: str):
        """Export to Tajima DST format."""
        data = bytearray()
        # DST header (125 bytes)
        header = b"LA:" + b" " * 122
        data.extend(header[:125])
        # Stitch data
        for block in plan.blocks:
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                if p1.jump or p2.jump:
                    data.extend(self._dst_jump(p2.x, p2.y))
                else:
                    data.extend(self._dst_stitch(p2.x - p1.x, p2.y - p1.y))
        # End
        data.extend(b"\x00\x00\xf3")
        Path(output_path).write_bytes(bytes(data))

    def _dst_stitch(self, dx: float, dy: float) -> bytes:
        """Encode a DST stitch command."""
        ix, iy = int(dx * 10), int(dy * 10)
        b0 = ((ix >> 1) & 0x07) | (((iy >> 1) & 0x07) << 3)
        b1 = ((ix >> 4) & 0x07) | (((iy >> 4) & 0x07) << 3) | (((ix >> 7) & 0x01) << 6) | (((iy >> 7) & 0x01) << 7)
        b2 = ((ix + 9) >> 8) & 0x03 | (((iy + 9) >> 8) & 0x03) << 2
        return bytes([b0 + 0x03, b1 + 0x83, b2 + 0xC3])

    def _dst_jump(self, x: float, y: float) -> bytes:
        """Encode a DST jump (move without stitching)."""
        return self._dst_stitch(x, y)  # Simplified

    def _export_pes(self, plan: StitchPlan, output_path: str):
        """Export to Brother PES format (simplified)."""
        # PES v1 header
        data = bytearray()
        data.extend(b"#PES0001")
        data.extend(b"\x00" * 4)  # PEC offset placeholder
        # Write stitch blocks as PEC section
        for block in plan.blocks:
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                dx = int(p2.x - p1.x)
                dy = int(p2.y - p1.y)
                # Clamp to signed byte range [-127, 127], then encode as unsigned byte
                dx_b = (max(-127, min(127, dx)) + 128) & 0xFF
                dy_b = (max(-127, min(127, dy)) + 128) & 0xFF
                data.extend(bytes([dx_b, dy_b]))
        data.extend(b"\xff\x00")  # End of design
        Path(output_path).write_bytes(bytes(data))

    def _export_exp(self, plan: StitchPlan, output_path: str):
        """Export to Melco EXP format."""
        data = bytearray()
        for block in plan.blocks:
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                dx = int(p2.x - p1.x)
                dy = int(p2.y - p1.y)
                dx_b = (max(-127, min(127, dx)) + 128) & 0xFF
                dy_b = (max(-127, min(127, dy)) + 128) & 0xFF
                data.extend(bytes([dx_b, dy_b]))
        data.extend(b"\x80\x00")  # End
        Path(output_path).write_bytes(bytes(data))

    def generate_preview_svg(self, plan: StitchPlan, output_path: str):
        """Alias for SVG export."""
        self._export_svg(plan, output_path)
