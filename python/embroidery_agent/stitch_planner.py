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
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            design_w = (max_x - min_x) / self.density
            design_h = (max_y - min_y) / self.density
        else:
            design_w = design_h = 0.0

        return StitchPlan(
            blocks=blocks,
            total_stitches=total_stitches,
            total_colors=unique_colors,
            design_width_mm=design_w,
            design_height_mm=design_h,
        )

    def _generate_stitches(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate stitches for a region based on its stitch type."""
        generators = {
            StitchType.RUNNING: self._gen_running,
            StitchType.SATIN: self._gen_satin,
            StitchType.FILL: self._gen_fill,
            StitchType.TATAMI: self._gen_tatami,
            StitchType.FRENCH_KNOT: self._gen_french_knot,
            StitchType.ZIGZAG: self._gen_zigzag,
            StitchType.CHAIN: self._gen_running,
            StitchType.SEED: self._gen_seed,
            StitchType.CROSS: self._gen_seed,
        }
        gen = generators.get(region.stitch_type, self._gen_fill)
        points = gen(region)
        return self._optimize_path(points)

    def _gen_running(self, region: ImageRegion) -> List[StitchPoint]:
        """Running stitch along contour."""
        if not region.contour:
            return []
        # Subsample contour for reasonable stitch count
        contour = self._subsample_points(region.contour, max_points=500)
        return [
            StitchPoint(x=float(x), y=float(y), stitch_type=StitchType.RUNNING, color=region.color)
            for x, y in contour
        ]

    def _gen_satin(self, region: ImageRegion) -> List[StitchPoint]:
        """Satin stitch: zigzag between contour edges."""
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
        """Parallel line fill stitch, clipped to region mask."""
        x1, y1, x2, y2 = region.bbox
        angle = self._compute_fill_angle(region)
        angle_rad = np.radians(angle)
        spacing = max(2, int(10.0 / self.density))
        points = []

        mask = region.mask  # cropped to bbox, shape (bbox_h+1, bbox_w+1)
        if mask is None:
            # Fallback: fill entire bbox
            for offset in range(y1, y2, spacing):
                dx = int((offset - y1) / np.tan(angle_rad + 1e-8)) if abs(np.tan(angle_rad)) > 0.01 else 0
                sx, ex = max(x1, x1 + dx), min(x2, x2 + dx)
                if sx < ex:
                    points.append(StitchPoint(x=float(sx), y=float(offset), stitch_type=StitchType.FILL, color=region.color))
                    points.append(StitchPoint(x=float(ex), y=float(offset), stitch_type=StitchType.FILL, color=region.color))
            return points

        # Clip fill lines to mask
        mask_h, mask_w = mask.shape
        for row_idx in range(0, mask_h, spacing):
            row = mask[row_idx]
            col_indices = np.where(row)[0]
            if len(col_indices) == 0:
                continue
            # Find contiguous runs in this row
            runs = self._find_runs(col_indices)
            for start_col, end_col in runs:
                # Convert back to image coordinates
                sx = x1 + start_col
                ex = x1 + end_col
                sy = y1 + row_idx
                points.append(StitchPoint(x=float(sx), y=float(sy), stitch_type=StitchType.FILL, color=region.color))
                points.append(StitchPoint(x=float(ex), y=float(sy), stitch_type=StitchType.FILL, color=region.color))

        return points

    def _gen_tatami(self, region: ImageRegion) -> List[StitchPoint]:
        """Dense tatami fill with offset rows, clipped to region mask."""
        x1, y1, x2, y2 = region.bbox
        spacing = max(2, int(8.0 / self.density))
        points = []

        mask = region.mask
        if mask is None:
            row = 0
            for offset in range(y1, y2, spacing):
                if row % 2 == 1:
                    sx, ex = x2, x1
                else:
                    sx, ex = x1, x2
                points.append(StitchPoint(x=float(sx), y=float(offset), stitch_type=StitchType.TATAMI, color=region.color))
                points.append(StitchPoint(x=float(ex), y=float(offset), stitch_type=StitchType.TATAMI, color=region.color))
                row += 1
            return points

        mask_h, mask_w = mask.shape
        row = 0
        for row_idx in range(0, mask_h, spacing):
            mask_row = mask[row_idx]
            col_indices = np.where(mask_row)[0]
            if len(col_indices) == 0:
                row += 1
                continue

            runs = self._find_runs(col_indices)
            for start_col, end_col in runs:
                sx = x1 + start_col
                ex = x1 + end_col
                sy = y1 + row_idx
                if row % 2 == 1:
                    sx, ex = ex, sx  # reverse direction
                points.append(StitchPoint(x=float(sx), y=float(sy), stitch_type=StitchType.TATAMI, color=region.color))
                points.append(StitchPoint(x=float(ex), y=float(sy), stitch_type=StitchType.TATAMI, color=region.color))
            row += 1

        return points

    def _gen_french_knot(self, region: ImageRegion) -> List[StitchPoint]:
        """Single point French knots."""
        cx, cy = region.centroid
        return [StitchPoint(x=float(cx), y=float(cy), stitch_type=StitchType.FRENCH_KNOT, color=region.color)]

    def _gen_zigzag(self, region: ImageRegion) -> List[StitchPoint]:
        """Zigzag decorative stitch along contour."""
        if not region.contour:
            return []
        contour = self._subsample_points(region.contour, max_points=200)
        points = []
        amplitude = 3.0
        for i, (x, y) in enumerate(contour):
            offset = amplitude if i % 2 == 0 else -amplitude
            points.append(StitchPoint(x=float(x + offset), y=float(y), stitch_type=StitchType.ZIGZAG, color=region.color))
        return points

    def _gen_seed(self, region: ImageRegion) -> List[StitchPoint]:
        """Scattered seed stitches within region mask."""
        x1, y1, x2, y2 = region.bbox
        mask = region.mask
        points = []

        if mask is not None:
            mask_h, mask_w = mask.shape
            ys, xs = np.where(mask)
            if len(ys) == 0:
                return []
            # Randomly sample points within mask
            rng = np.random.RandomState(region.region_id)
            n_seeds = min(50, len(ys))
            indices = rng.choice(len(ys), n_seeds, replace=False)
            for idx in indices:
                px = x1 + int(xs[idx])
                py = y1 + int(ys[idx])
                points.append(StitchPoint(x=float(px), y=float(py), stitch_type=StitchType.SEED, color=region.color))
        else:
            cx, cy = region.centroid
            points.append(StitchPoint(x=float(cx), y=float(cy), stitch_type=StitchType.SEED, color=region.color))

        return points

    @staticmethod
    def _find_runs(col_indices: np.ndarray, gap: int = 2) -> List[Tuple[int, int]]:
        """Find contiguous runs of column indices."""
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

    @staticmethod
    def _subsample_points(points: List[Tuple[int, int]], max_points: int = 500) -> List[Tuple[int, int]]:
        """Subsample a list of points to at most max_points."""
        if len(points) <= max_points:
            return points
        step = len(points) / max_points
        return [points[int(i * step)] for i in range(max_points)]

    def _optimize_path(self, points: List[StitchPoint]) -> List[StitchPoint]:
        """Nearest-neighbor TSP approximation for stitch path optimization."""
        if len(points) <= 2:
            return points
        remaining = list(points)
        sorted_pts = [remaining.pop(0)]
        while remaining:
            last = sorted_pts[-1]
            min_idx = min(range(len(remaining)), key=lambda i: (remaining[i].x - last.x) ** 2 + (remaining[i].y - last.y) ** 2)
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
