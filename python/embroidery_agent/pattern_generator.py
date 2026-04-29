"""
Pattern generator module for Embroidery Agent.

Generates standard embroidery machine file formats:
    - PES (Brother/Baby Lock)
    - DST (Tajima)
    - EXP (Melco)
    - SVG (preview)
    - PNG (preview for Streamlit)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw

from .image_processor import EmbroideryColor, StitchType, ImageRegion
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
        """Generate SVG preview of stitch plan.

        Renders stitch lines with proper coordinate mapping.
        The stitch points are in pixel coordinates; we map them to SVG
        coordinates preserving the original aspect ratio.
        """
        if not plan.blocks:
            Path(output_path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect width="100%" height="100%" fill="white"/></svg>')
            return

        # Compute bounding box of all stitch points (in pixel coords)
        all_points = [p for b in plan.blocks for p in b.points]
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)

        pixel_w = max(max_x - min_x, 1)
        pixel_h = max(max_y - min_y, 1)

        # SVG canvas: fit into 400x400 max, preserving aspect ratio
        max_svg = 400
        margin = 10
        usable = max_svg - 2 * margin
        scale = min(usable / pixel_w, usable / pixel_h)
        svg_w = int(pixel_w * scale) + 2 * margin
        svg_h = int(pixel_h * scale) + 2 * margin

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
            f'viewBox="0 0 {svg_w} {svg_h}">',
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
                x1 = margin + (p1.x - min_x) * scale
                y1 = margin + (p1.y - min_y) * scale
                x2 = margin + (p2.x - min_x) * scale
                y2 = margin + (p2.y - min_y) * scale
                lines.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{color_hex}" stroke-width="0.5" opacity="0.8"/>'
                )

        lines.append("</svg>")
        Path(output_path).write_text("\n".join(lines))

    def generate_preview_svg(self, plan: StitchPlan, output_path: str,
                             regions: Optional[List[ImageRegion]] = None):
        """Generate enhanced SVG preview with region fills and stitch lines.

        If regions are provided, renders colored region shapes first,
        then overlays stitch lines for a more realistic preview.
        """
        if not plan.blocks and not regions:
            Path(output_path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect width="100%" height="100%" fill="white"/></svg>')
            return

        # Compute bounding box from regions or stitch points
        if regions:
            all_x = []
            all_y = []
            for r in regions:
                all_x.extend([r.bbox[0], r.bbox[2]])
                all_y.extend([r.bbox[1], r.bbox[3]])
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
        else:
            all_points = [p for b in plan.blocks for p in b.points]
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)

        pixel_w = max(max_x - min_x, 1)
        pixel_h = max(max_y - min_y, 1)

        max_svg = 500
        margin = 10
        usable = max_svg - 2 * margin
        scale = min(usable / pixel_w, usable / pixel_h)
        svg_w = int(pixel_w * scale) + 2 * margin
        svg_h = int(pixel_h * scale) + 2 * margin

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
            f'viewBox="0 0 {svg_w} {svg_h}">',
            f'<rect width="100%" height="100%" fill="#f8f8f8"/>',
        ]

        # Render region fills from masks
        if regions:
            for region in regions:
                if region.mask is None:
                    continue
                color_hex = region.color.hex if region.color else "#cccccc"
                mask_h, mask_w = region.mask.shape
                rx1, ry1 = region.bbox[0], region.bbox[1]

                # Convert mask to SVG polygon paths (simplified: render filled rectangles per row)
                for row_idx in range(0, mask_h, 2):  # skip every other row for performance
                    row = region.mask[row_idx]
                    col_indices = np.where(row)[0]
                    if len(col_indices) == 0:
                        continue
                    # Find contiguous runs
                    runs = self._find_mask_runs(col_indices)
                    for start_col, end_col in runs:
                        sx = margin + (rx1 + start_col - min_x) * scale
                        sy = margin + (ry1 + row_idx - min_y) * scale
                        sw = (end_col - start_col + 1) * scale
                        sh = max(2 * scale, 1)
                        lines.append(
                            f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" '
                            f'fill="{color_hex}" opacity="0.6"/>'
                        )

        # Overlay stitch lines
        for block in plan.blocks:
            if not block.points:
                continue
            color_hex = block.color.hex if block.color else "#000000"
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                if p1.jump or p2.jump:
                    continue
                x1 = margin + (p1.x - min_x) * scale
                y1 = margin + (p1.y - min_y) * scale
                x2 = margin + (p2.x - min_x) * scale
                y2 = margin + (p2.y - min_y) * scale
                lines.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{color_hex}" stroke-width="0.3" opacity="0.5"/>'
                )

        lines.append("</svg>")
        Path(output_path).write_text("\n".join(lines))

    @staticmethod
    def _find_mask_runs(col_indices: np.ndarray, gap: int = 2) -> List[Tuple[int, int]]:
        """Find contiguous runs of column indices in a mask row."""
        if len(col_indices) == 0:
            return []
        runs = []
        start = int(col_indices[0])
        prev = start
        for c in col_indices[1:]:
            c = int(c)
            if c - prev > gap:
                runs.append((start, prev))
                start = c
            prev = c
        runs.append((start, prev))
        return runs

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
                dx, dy = int(p2.x - p1.x), int(p2.y - p1.y)
                data.extend(bytes([max(-127, min(127, dx)), max(-127, min(127, dy))]))
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
                data.extend(bytes([max(-127, min(127, dx)), max(-127, min(127, dy))]))
        data.extend(b"\x80\x00")  # End
        Path(output_path).write_bytes(bytes(data))

    def generate_preview_png(self, plan: StitchPlan, output_path: str,
                             regions: Optional[List[ImageRegion]] = None,
                             max_size: int = 500):
        """Generate PNG preview with region fills and stitch lines.

        Uses Pillow for rendering — guaranteed to work with st.image().
        """
        # Compute bounding box
        if regions:
            all_x, all_y = [], []
            for r in regions:
                all_x.extend([r.bbox[0], r.bbox[2]])
                all_y.extend([r.bbox[1], r.bbox[3]])
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
        elif plan.blocks:
            all_points = [p for b in plan.blocks for p in b.points]
            if not all_points:
                # Empty — create placeholder
                img = Image.new("RGB", (200, 200), (248, 248, 248))
                img.save(output_path)
                return
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
        else:
            img = Image.new("RGB", (200, 200), (248, 248, 248))
            img.save(output_path)
            return

        pixel_w = max(max_x - min_x, 1)
        pixel_h = max(max_y - min_y, 1)

        margin = 10
        usable = max_size - 2 * margin
        scale = min(usable / pixel_w, usable / pixel_h)
        img_w = int(pixel_w * scale) + 2 * margin
        img_h = int(pixel_h * scale) + 2 * margin

        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw region fills
        if regions:
            for region in regions:
                if region.mask is None:
                    continue
                color_rgb = region.color.rgb if region.color else (200, 200, 200)
                mask_h, mask_w = region.mask.shape
                rx1, ry1 = region.bbox[0], region.bbox[1]

                # Render mask as filled pixels (downsample for performance)
                step = max(1, int(1 / scale))
                for row_idx in range(0, mask_h, step):
                    row = region.mask[row_idx]
                    col_indices = np.where(row)[0]
                    if len(col_indices) == 0:
                        continue
                    runs = self._find_mask_runs(col_indices)
                    for start_col, end_col in runs:
                        sx = int(margin + (rx1 + start_col - min_x) * scale)
                        sy = int(margin + (ry1 + row_idx - min_y) * scale)
                        ex = int(margin + (rx1 + end_col + 1 - min_x) * scale)
                        ey = int(margin + (ry1 + row_idx + step - min_y) * scale)
                        draw.rectangle([sx, sy, ex, ey], fill=color_rgb)

        # Draw stitch lines
        for block in plan.blocks:
            if not block.points:
                continue
            color_rgb = block.color.rgb if block.color else (0, 0, 0)
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                if p1.jump or p2.jump:
                    continue
                x1 = margin + (p1.x - min_x) * scale
                y1 = margin + (p1.y - min_y) * scale
                x2 = margin + (p2.x - min_x) * scale
                y2 = margin + (p2.y - min_y) * scale
                draw.line([(x1, y1), (x2, y2)], fill=color_rgb, width=1)

        img.save(output_path)
