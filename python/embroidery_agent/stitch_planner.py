"""
Stitch path planning module for Embroidery Agent.

Converts image regions into ordered stitch point sequences.
Handles:
    - Running stitch: follow contours
    - Satin stitch: zigzag between contour pairs
    - Fill stitch: parallel line fill with angle optimization
    - Tatami stitch: dense fill with offset rows
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

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


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
        self.density = density        # stitches per mm
        self.stitch_length = stitch_length  # mm between stitches

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
            xs = [p.x for p in all_points]
            ys = [p.y for p in all_points]
            w_mm = (max(xs) - min(xs)) / 10.0
            h_mm = (max(ys) - min(ys)) / 10.0
        else:
            w_mm, h_mm = 0.0, 0.0

        return StitchPlan(
            blocks=blocks,
            total_stitches=total_stitches,
            total_colors=unique_colors,
            design_width_mm=w_mm,
            design_height_mm=h_mm,
        )

    def _generate_stitches(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate stitch points based on region's assigned stitch type."""
        generator = {
            StitchType.RUNNING: self._gen_running,
            StitchType.SATIN: self._gen_satin,
            StitchType.FILL: self._gen_fill,
            StitchType.TATAMI: self._gen_tatami,
            StitchType.FRENCH_KNOT: self._gen_french_knot,
            StitchType.CHAIN: self._gen_running,
            StitchType.ZIGZAG: self._gen_zigzag,
            StitchType.CROSS: self._gen_cross,
            StitchType.SEED: self._gen_seed,
        }
        gen_fn = generator.get(region.stitch_type, self._gen_fill)
        return gen_fn(region)

    def _gen_running(self, region: ImageRegion) -> List[StitchPoint]:
        """Follow contour with running stitch."""
        if not region.contour:
            return []
        contour = self._optimize_path(region.contour)
        step = max(1, len(contour) // max(int(len(contour) * self.density / 10), 1))
        points = []
        for i in range(0, len(contour), step):
            x, y = contour[i]
            points.append(StitchPoint(x=float(x), y=float(y), stitch_type=StitchType.RUNNING, color=region.color))
        return points

    def _gen_satin(self, region: ImageRegion) -> List[StitchPoint]:
        """Zigzag satin stitch between contour edges."""
        if len(region.contour) < 4:
            return self._gen_running(region)
        left, right = self._split_contour_pair(region.contour)
        if not left or not right:
            return self._gen_running(region)
        points = []
        n = min(len(left), len(right))
        for i in range(n):
            points.append(StitchPoint(x=float(left[i][0]), y=float(left[i][1]), stitch_type=StitchType.SATIN, color=region.color))
            points.append(StitchPoint(x=float(right[i][0]), y=float(right[i][1]), stitch_type=StitchType.SATIN, color=region.color))
        return points

    def _gen_fill(self, region: ImageRegion) -> List[StitchPoint]:
        """Parallel line fill stitch."""
        x1, y1, x2, y2 = region.bbox
        angle = self._compute_fill_angle(region)
        angle_rad = np.radians(angle)
        spacing = max(1, int(10.0 / self.density))
        points = []
        for offset in range(y1, y2, spacing):
            dx = int((offset - y1) / np.tan(angle_rad + 1e-8)) if abs(np.tan(angle_rad)) > 0.01 else 0
            sx, ex = max(x1, x1 + dx), min(x2, x2 + dx)
            if sx < ex:
                points.append(StitchPoint(x=float(sx), y=float(offset), stitch_type=StitchType.FILL, color=region.color))
                points.append(StitchPoint(x=float(ex), y=float(offset), stitch_type=StitchType.FILL, color=region.color))
        return points

    def _gen_tatami(self, region: ImageRegion) -> List[StitchPoint]:
        """Dense tatami fill with offset rows."""
        x1, y1, x2, y2 = region.bbox
        spacing = max(1, int(8.0 / self.density))
        points = []
        row = 0
        for offset in range(y1, y2, spacing):
            if row % 2 == 1:
                sx, ex = x2, x1  # reverse direction
            else:
                sx, ex = x1, x2
            points.append(StitchPoint(x=float(sx), y=float(offset), stitch_type=StitchType.TATAMI, color=region.color))
            points.append(StitchPoint(x=float(ex), y=float(offset), stitch_type=StitchType.TATAMI, color=region.color))
            row += 1
        return points

    def _gen_french_knot(self, region: ImageRegion) -> List[StitchPoint]:
        """Single point French knots."""
        cx, cy = region.centroid
        return [StitchPoint(x=float(cx), y=float(cy), stitch_type=StitchType.FRENCH_KNOT, color=region.color)]

    def _gen_zigzag(self, region: ImageRegion) -> List[StitchPoint]:
        """Zigzag decorative stitch."""
        x1, y1, x2, y2 = region.bbox
        cx = (x1 + x2) / 2
        amplitude = (x2 - x1) / 4
        points = []
        for y in range(y1, y2, max(1, int(6.0 / self.density))):
            points.append(StitchPoint(x=cx - amplitude, y=float(y), stitch_type=StitchType.ZIGZAG, color=region.color))
            points.append(StitchPoint(x=cx + amplitude, y=float(y), stitch_type=StitchType.ZIGZAG, color=region.color))
        return points

    def _gen_cross(self, region: ImageRegion) -> List[StitchPoint]:
        """Cross stitch pattern."""
        x1, y1, x2, y2 = region.bbox
        size = max(2, int(10.0 / self.density))
        points = []
        for y in range(y1, y2, size):
            for x in range(x1, x2, size):
                points.append(StitchPoint(x=float(x), y=float(y), stitch_type=StitchType.CROSS, color=region.color))
                points.append(StitchPoint(x=float(x + size), y=float(y + size), stitch_type=StitchType.CROSS, color=region.color))
                points.append(StitchPoint(x=float(x + size), y=float(y), stitch_type=StitchType.CROSS, color=region.color))
                points.append(StitchPoint(x=float(x), y=float(y + size), stitch_type=StitchType.CROSS, color=region.color))
        return points

    def _gen_seed(self, region: ImageRegion) -> List[StitchPoint]:
        """Scattered seed stitch."""
        x1, y1, x2, y2 = region.bbox
        rng = np.random.RandomState(region.region_id)
        n = max(3, int(region.area / 500))
        points = []
        for _ in range(n):
            x = rng.randint(x1, x2)
            y = rng.randint(y1, y2)
            points.append(StitchPoint(x=float(x), y=float(y), stitch_type=StitchType.SEED, color=region.color))
        return points

    def _optimize_path(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Nearest-neighbor TSP approximation for path optimization."""
        if len(points) <= 2:
            return points
        remaining = list(points)
        sorted_pts = [remaining.pop(0)]
        while remaining:
            last = sorted_pts[-1]
            min_idx = min(range(len(remaining)), key=lambda i: (remaining[i][0] - last[0]) ** 2 + (remaining[i][1] - last[1]) ** 2)
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
