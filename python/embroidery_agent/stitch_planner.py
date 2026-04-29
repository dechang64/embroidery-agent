"""
Stitch path planning module for Embroidery Agent.

Converts image regions into ordered stitch point sequences.
Handles:
    - Running stitch: follow contours
    - Satin stitch: zigzag between contour pairs
    - Fill stitch: parallel line fill clipped to region mask
    - Tatami stitch: dense fill with offset rows clipped to region mask
    - French knot: single points
    - Path optimization: nearest-neighbor TSP approximation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .image_processor import StitchType, ImageRegion, EmbroideryColor


@dataclass
class StitchPoint:
    """A single stitch point with position and attributes."""
    x: float
    y: float
    stitch_type: StitchType = StitchType.RUNNING
    color: Optional[EmbroideryColor] = None
    jump: bool = False
    trim: bool = False
    color_change: bool = False

    @property
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class StitchBlock:
    """A contiguous block of stitches with the same color and type."""
    stitch_type: StitchType
    color: EmbroideryColor
    points: List[StitchPoint] = field(default_factory=list)
    stitch_count: int = 0


@dataclass
class StitchPlan:
    """Complete stitch plan for a design."""
    blocks: List[StitchBlock] = field(default_factory=list)
    total_stitches: int = 0
    total_colors: int = 0
    design_width_mm: float = 0.0
    design_height_mm: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class StitchPlanner:
    """Converts processed image regions into stitch plans."""

    def __init__(self, density: float = 4.0, stitch_length: float = 2.5):
        self.density = density
        self.stitch_length = stitch_length

    def plan(self, regions: List[ImageRegion],
             color_palette: List[EmbroideryColor]) -> StitchPlan:
        """Generate complete stitch plan from regions."""
        blocks = []
        for region in sorted(regions, key=lambda r: r.priority):
            color = region.color or (color_palette[0] if color_palette else EmbroideryColor("black", (0, 0, 0)))
            points = self._generate_stitches(region)
            if points:
                block = StitchBlock(
                    stitch_type=region.stitch_type,
                    color=color,
                    points=points,
                    stitch_count=len(points),
                )
                blocks.append(block)

        total_stitches = sum(b.stitch_count for b in blocks)
        unique_colors = len(set(b.color.hex for b in blocks))

        # Compute design bounds
        all_points = [p for b in blocks for p in b.points]
        if all_points:
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            w_mm = (max_x - min_x) / 10.0
            h_mm = (max_y - min_y) / 10.0
        else:
            w_mm = h_mm = 0.0

        return StitchPlan(
            blocks=blocks,
            total_stitches=total_stitches,
            total_colors=unique_colors,
            design_width_mm=w_mm,
            design_height_mm=h_mm,
        )

    def _generate_stitches(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate stitches for a region based on its stitch type."""
        dispatch = {
            StitchType.RUNNING: self._gen_running,
            StitchType.SATIN: self._gen_satin,
            StitchType.FILL: self._gen_fill,
            StitchType.TATAMI: self._gen_tatami,
            StitchType.FRENCH_KNOT: self._gen_french_knot,
            StitchType.ZIGZAG: self._gen_zigzag,
            StitchType.CHAIN: self._gen_running,
            StitchType.CROSS: self._gen_fill,
            StitchType.SEED: self._gen_seed,
        }
        gen_fn = dispatch.get(region.stitch_type, self._gen_fill)
        points = gen_fn(region)
        return self._optimize_path(points)

    def _gen_running(self, region: ImageRegion) -> List[StitchPoint]:
        """Running stitch along contour."""
        if not region.contour:
            return []
        # Subsample contour for reasonable stitch count
        contour = self._subsample(region.contour, max_points=200)
        return [StitchPoint(x=float(p[0]), y=float(p[1]),
                            stitch_type=StitchType.RUNNING, color=region.color)
                for p in contour]

    def _gen_satin(self, region: ImageRegion) -> List[StitchPoint]:
        """Satin stitch: zigzag between contour edges."""
        if len(region.contour) < 4:
            return self._gen_running(region)
        left, right = self._split_contour_pair(region.contour)
        if not left or not right:
            return self._gen_running(region)
        points = []
        n = min(len(left), len(right))
        for i in range(n):
            points.append(StitchPoint(x=float(left[i][0]), y=float(left[i][1]),
                                      stitch_type=StitchType.SATIN, color=region.color))
            points.append(StitchPoint(x=float(right[i][0]), y=float(right[i][1]),
                                      stitch_type=StitchType.SATIN, color=region.color))
        return points

    def _gen_fill(self, region: ImageRegion) -> List[StitchPoint]:
        """Parallel line fill stitch, clipped to region mask."""
        x1, y1, x2, y2 = region.bbox
        angle = self._compute_fill_angle(region)
        angle_rad = np.radians(angle)
        spacing = max(1, int(10.0 / self.density))
        points = []

        for offset in range(y1, y2, spacing):
            dx = int((offset - y1) / np.tan(angle_rad + 1e-8)) if abs(np.tan(angle_rad)) > 0.01 else 0
            sx, ex = max(x1, x1 + dx), min(x2, x2 + dx)
            if sx < ex:
                # Clip to mask if available
                if region.mask is not None:
                    local_y = offset - y1
                    if 0 <= local_y < region.mask.shape[0]:
                        row = region.mask[local_y]
                        # Find actual filled range in this row
                        cols = np.where(row)[0]
                        if len(cols) == 0:
                            continue
                        local_sx = max(int(sx - x1), int(cols[0]))
                        local_ex = min(int(ex - x1), int(cols[-1]))
                        if local_sx >= local_ex:
                            continue
                        sx = x1 + local_sx
                        ex = x1 + local_ex
                    else:
                        continue

                points.append(StitchPoint(x=float(sx), y=float(offset),
                                          stitch_type=StitchType.FILL, color=region.color))
                points.append(StitchPoint(x=float(ex), y=float(offset),
                                          stitch_type=StitchType.FILL, color=region.color))
        return points

    def _gen_tatami(self, region: ImageRegion) -> List[StitchPoint]:
        """Dense tatami fill with offset rows, clipped to region mask."""
        x1, y1, x2, y2 = region.bbox
        spacing = max(1, int(8.0 / self.density))
        points = []
        row = 0
        for offset in range(y1, y2, spacing):
            # Clip to mask
            if region.mask is not None:
                local_y = offset - y1
                if 0 <= local_y < region.mask.shape[0]:
                    row_mask = region.mask[local_y]
                    cols = np.where(row_mask)[0]
                    if len(cols) == 0:
                        row += 1
                        continue
                    sx_local, ex_local = int(cols[0]), int(cols[-1])
                else:
                    row += 1
                    continue
            else:
                sx_local, ex_local = 0, x2 - x1

            if row % 2 == 1:
                sx, ex = x1 + ex_local, x1 + sx_local
            else:
                sx, ex = x1 + sx_local, x1 + ex_local

            if sx < ex:
                points.append(StitchPoint(x=float(sx), y=float(offset),
                                          stitch_type=StitchType.TATAMI, color=region.color))
                points.append(StitchPoint(x=float(ex), y=float(offset),
                                          stitch_type=StitchType.TATAMI, color=region.color))
            row += 1
        return points

    def _gen_french_knot(self, region: ImageRegion) -> List[StitchPoint]:
        """Single point French knots."""
        cx, cy = region.centroid
        return [StitchPoint(x=float(cx), y=float(cy),
                            stitch_type=StitchType.FRENCH_KNOT, color=region.color)]

    def _gen_zigzag(self, region: ImageRegion) -> List[StitchPoint]:
        """Zigzag decorative stitch."""
        x1, y1, x2, y2 = region.bbox
        cx = (x1 + x2) / 2
        amplitude = (x2 - x1) / 4
        points = []
        for y in range(y1, y2, max(1, int(5.0 / self.density))):
            # Clip to mask
            if region.mask is not None:
                local_y = y - y1
                if 0 <= local_y < region.mask.shape[0]:
                    cols = np.where(region.mask[local_y])[0]
                    if len(cols) == 0:
                        continue
                    x1c, x2c = x1 + int(cols[0]), x1 + int(cols[-1])
                else:
                    continue
            else:
                x1c, x2c = x1, x2

            cx_local = (x1c + x2c) / 2
            amp = (x2c - x1c) / 4
            points.append(StitchPoint(x=cx_local - amp, y=float(y),
                                      stitch_type=StitchType.ZIGZAG, color=region.color))
            points.append(StitchPoint(x=cx_local + amp, y=float(y),
                                      stitch_type=StitchType.ZIGZAG, color=region.color))
        return points

    def _gen_seed(self, region: ImageRegion) -> List[StitchPoint]:
        """Scattered seed stitches within region."""
        if region.mask is None:
            return []
        mask_h, mask_w = region.mask.shape
        rx1, ry1 = region.bbox[0], region.bbox[1]
        ys, xs = np.where(region.mask)
        if len(xs) == 0:
            return []
        # Random sample
        rng = np.random.RandomState(42)
        n = min(50, len(xs))
        indices = rng.choice(len(xs), n, replace=False)
        return [StitchPoint(x=float(rx1 + xs[i]), y=float(ry1 + ys[i]),
                            stitch_type=StitchType.SEED, color=region.color)
                for i in indices]

    @staticmethod
    def _subsample(points: List[Tuple[int, int]], max_points: int = 200) -> List[Tuple[int, int]]:
        """Subsample a list of points to max_points."""
        if len(points) <= max_points:
            return points
        step = len(points) / max_points
        return [points[int(i * step)] for i in range(max_points)]

    def _optimize_path(self, points: List[StitchPoint]) -> List[StitchPoint]:
        """Nearest-neighbor TSP approximation."""
        if len(points) <= 2:
            return points
        remaining = list(points)
        sorted_pts = [remaining.pop(0)]
        while remaining:
            last = sorted_pts[-1]
            min_idx = min(range(len(remaining)),
                          key=lambda i: (remaining[i].x - last.x) ** 2 + (remaining[i].y - last.y) ** 2)
            sorted_pts.append(remaining.pop(min_idx))
        return sorted_pts

    def _split_contour_pair(self, contour: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Split contour into left and right edges for satin stitch."""
        if len(contour) < 4:
            return [], []
        sorted_by_y = sorted(contour, key=lambda p: (p[1], p[0]))
        mid = len(sorted_by_y) // 2
        left = sorted(sorted_by_y[:mid], key=lambda p: p[1])
        right = sorted(sorted_by_y[mid:], key=lambda p: p[1])
        return left, right

    def _compute_fill_angle(self, region: ImageRegion) -> float:
        """Compute optimal fill angle based on region shape."""
        bbox = region.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > h * 2:
            return 90
        elif h > w * 2:
            return 0
        else:
            return 45
