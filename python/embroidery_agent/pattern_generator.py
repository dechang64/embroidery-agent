"""
Pattern generator module for Embroidery Agent.

Generates standard embroidery machine file formats:
    - PES (Brother/Baby Lock)
    - DST (Tajima)
    - EXP (Melco)
    - SVG (preview)
    - PNG (preview for Streamlit — uses quantized image + stitch texture overlay)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

from .image_processor import EmbroideryColor, StitchType, ImageRegion, ProcessedImage
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
        self.dpi = dpi

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
            format=fmt, file_path=path,
            stitch_count=plan.total_stitches, color_count=plan.total_colors,
            design_width_mm=plan.design_width_mm, design_height_mm=plan.design_height_mm,
            file_size_bytes=file_size,
        )

    def _export_svg(self, plan: StitchPlan, output_path: str):
        """Generate SVG preview from stitch points."""
        all_points = [p for b in plan.blocks for p in b.points]
        if not all_points:
            Path(output_path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect width="100%" height="100%" fill="white"/></svg>')
            return

        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)

        pixel_w = max(max_x - min_x, 1)
        pixel_h = max(max_y - min_y, 1)

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
        """Alias — delegates to _export_svg."""
        self._export_svg(plan, output_path)

    def generate_preview_png(self, plan: StitchPlan, output_path: str,
                             regions: Optional[List[ImageRegion]] = None,
                             processed: Optional[ProcessedImage] = None,
                             max_size: int = 500):
        """Generate PNG preview.

        Strategy: use the color-quantized image as base, then overlay
        stitch texture lines for an embroidery look.
        """
        # Use quantized image as base if available (best visual result)
        if processed and processed.quantized_image:
            base_img = processed.quantized_image.copy()
        elif regions:
            # Reconstruct from regions + masks
            orig_w, orig_h = 200, 200  # fallback
            if regions:
                orig_w = max(r.bbox[2] for r in regions) + 1
                orig_h = max(r.bbox[3] for r in regions) + 1
            base_img = Image.new("RGB", (orig_w, orig_h), (255, 255, 255))
            draw = ImageDraw.Draw(base_img)
            for region in regions:
                if region.mask is None:
                    continue
                color_rgb = region.color.rgb if region.color else (200, 200, 200)
                mask_h, mask_w = region.mask.shape
                rx1, ry1 = region.bbox[0], region.bbox[1]
                for row_idx in range(mask_h):
                    row = region.mask[row_idx]
                    cols = np.where(row)[0]
                    if len(cols) == 0:
                        continue
                    runs = self._find_mask_runs(cols)
                    for sc, ec in runs:
                        draw.rectangle(
                            [rx1 + sc, ry1 + row_idx, rx1 + ec, ry1 + row_idx],
                            fill=color_rgb,
                        )
        else:
            # Fallback: draw from stitch points only
            all_points = [p for b in plan.blocks for p in b.points]
            if not all_points:
                Image.new("RGB", (200, 200), (248, 248, 248)).save(output_path)
                return
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            w = int(max_x - min_x) + 1
            h = int(max_y - min_y) + 1
            base_img = Image.new("RGB", (w, h), (255, 255, 255))
            draw = ImageDraw.Draw(base_img)
            for block in plan.blocks:
                if not block.points:
                    continue
                color_rgb = block.color.rgb if block.color else (0, 0, 0)
                for i in range(1, len(block.points)):
                    p1, p2 = block.points[i - 1], block.points[i]
                    if p1.jump or p2.jump:
                        continue
                    draw.line([(p1.x - min_x, p1.y - min_y), (p2.x - min_x, p2.y - min_y)],
                              fill=color_rgb, width=1)

        # Resize to max_size
        orig_w, orig_h = base_img.size
        if max(orig_w, orig_h) > max_size:
            ratio = max_size / max(orig_w, orig_h)
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            base_img = base_img.resize((new_w, new_h), Image.LANCZOS)

        # Overlay stitch texture: diagonal hatch lines for embroidery feel
        texture = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        tex_draw = ImageDraw.Draw(texture)
        w, h = base_img.size
        spacing = 6
        for i in range(-h, w + h, spacing):
            tex_draw.line([(i, 0), (i + h, h)], fill=(0, 0, 0, 25), width=1)
            tex_draw.line([(i, h), (i + h, 0)], fill=(0, 0, 0, 15), width=1)

        # Composite: base image + texture
        result = base_img.convert("RGBA")
        result = Image.alpha_composite(result, texture)

        # Convert back to RGB and save
        result = result.convert("RGB")
        result.save(output_path)

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
        header = b"LA:" + b" " * 122
        data.extend(header[:125])
        for block in plan.blocks:
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                if p1.jump or p2.jump:
                    data.extend(self._dst_jump(p2.x, p2.y))
                else:
                    data.extend(self._dst_stitch(p2.x, p2.y))
        data.extend(b"\x00\x00")  # End
        Path(output_path).write_bytes(bytes(data))

    def _dst_stitch(self, x: float, y: float) -> bytes:
        """Encode a DST stitch."""
        ix, iy = int(x), int(y)
        ix = ((ix + 9) >> 3) & 0x0F | (((ix + 9) >> 8) & 0x03) << 2
        iy = ((iy + 9) >> 3) & 0x03 | (((iy + 9) >> 8) & 0x03) << 2
        return bytes([ix + 0x03, iy + 0x83, 0xC3])

    def _dst_jump(self, x: float, y: float) -> bytes:
        return self._dst_stitch(x, y)

    def _export_pes(self, plan: StitchPlan, output_path: str):
        """Export to Brother PES format (simplified)."""
        data = bytearray()
        data.extend(b"#PES0001")
        data.extend(b"\x00" * 4)
        for block in plan.blocks:
            for i in range(1, len(block.points)):
                p1, p2 = block.points[i - 1], block.points[i]
                dx, dy = int(p2.x - p1.x), int(p2.y - p1.y)
                data.extend(bytes([max(-127, min(127, dx)), max(-127, min(127, dy))]))
        data.extend(b"\xff\x00")
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
        data.extend(b"\x80\x00")
        Path(output_path).write_bytes(bytes(data))
