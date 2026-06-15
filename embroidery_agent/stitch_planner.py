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

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .image_processor import StitchType, ImageRegion, EmbroideryColor


@dataclass
class StitchPoint:
    """A single stitch point with position and attributes."""
    x: float          # mm
    y: float          # mm
    stitch_type: StitchType = StitchType.RUNNING
    color: Optional[EmbroideryColor] = None
    jump: bool = False     # True = needle up (move without stitching)
    trim: bool = False     # True = trim thread after this point
    color_change: bool = False  # True = change color at this point

    @property
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class StitchBlock:
    """A contiguous block of stitches with the same color and stitch type."""
    color: EmbroideryColor
    stitch_type: StitchType
    points: List[StitchPoint] = field(default_factory=list)
    stitch_count: int = 0

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class StitchPlan:
    """Complete stitch plan for an embroidery design."""
    blocks: List[StitchBlock] = field(default_factory=list)
    total_stitches: int = 0
    total_colors: int = 0
    design_width_mm: float = 0.0
    design_height_mm: float = 0.0

    def add_block(self, block: StitchBlock):
        self.blocks.append(block)
        self.total_stitches += block.stitch_count
        self.total_colors = len({b.color.name for b in self.blocks})

    def get_all_points(self) -> List[StitchPoint]:
        """Flatten all blocks into a single point sequence."""
        points = []
        for block in self.blocks:
            points.extend(block.points)
        return points


class StitchPlanner:
    """Plans stitch paths from image regions.

    Parameters:
        stitch_density: stitches per mm for fill patterns
        satin_width: width of satin stitches in mm
        running_step: step size for running stitch contour following in mm
        resolution: pixels per mm (for coordinate conversion)
    """

    def __init__(self, stitch_density: float = 3.0, satin_width: float = 3.0,
                 running_step: float = 0.5, resolution: float = 5.0):
        self.stitch_density = stitch_density
        self.satin_width = satin_width
        self.running_step = running_step
        self.resolution = resolution  # pixels per mm

    def plan(self, regions: List[ImageRegion],
             palette: List[EmbroideryColor]) -> StitchPlan:
        """Convert all regions into a complete stitch plan."""
        plan = StitchPlan()

        # Group regions by color
        color_groups: Dict[str, List[ImageRegion]] = {}
        for region in regions:
            color_key = str(region.dominant_color)
            if color_key not in color_groups:
                color_groups[color_key] = []
            color_groups[color_key].append(region)

        # Process each color group
        for color_idx, (color_key, group_regions) in enumerate(color_groups.items()):
            thread = palette[color_idx] if color_idx < len(palette) else palette[0]

            for region in group_regions:
                block = self._region_to_block(region, thread)
                if block and block.stitch_count > 0:
                    plan.add_block(block)

        # Calculate design dimensions
        if plan.blocks:
            all_points = plan.get_all_points()
            xs = [p.x for p in all_points]
            ys = [p.y for p in all_points]
            plan.design_width_mm = max(xs) - min(xs) if xs else 0
            plan.design_height_mm = max(ys) - min(ys) if ys else 0

        return plan

    def _region_to_block(self, region: ImageRegion,
                         thread: EmbroideryColor) -> Optional[StitchBlock]:
        """Convert a single region to a stitch block."""
        if region.stitch_type == StitchType.RUNNING:
            points = self._plan_running(region)
        elif region.stitch_type == StitchType.SATIN:
            points = self._plan_satin(region)
        elif region.stitch_type == StitchType.FILL:
            points = self._plan_fill(region)
        elif region.stitch_type == StitchType.TATAMI:
            points = self._plan_tatami(region)
        elif region.stitch_type == StitchType.FRENCH_KNOT:
            points = self._plan_french_knot(region)
        else:
            points = self._plan_fill(region)

        if not points:
            return None

        return StitchBlock(
            color=thread,
            stitch_type=region.stitch_type,
            points=points,
            stitch_count=len(points),
        )

    def _px_to_mm(self, px: float) -> float:
        """Convert pixel coordinates to mm."""
        return px / self.resolution

    def _plan_running(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate running stitch along contour."""
        if not region.contour:
            return []

        # Sort contour points for continuous path
        sorted_contour = self._sort_contour(region.contour)

        # Subsample for stitch density
        step_px = max(1, int(self.running_step * self.resolution))
        points = []
        for i in range(0, len(sorted_contour), step_px):
            x, y = sorted_contour[i]
            points.append(StitchPoint(
                x=self._px_to_mm(x),
                y=self._px_to_mm(y),
                stitch_type=StitchType.RUNNING,
            ))

        return points

    def _plan_satin(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate satin stitch (zigzag between contour edges)."""
        contour = region.contour
        if len(contour) < 4:
            return self._plan_running(region)

        # Split contour into left and right edges
        left, right = self._split_contour_pair(contour)

        if not left or not right:
            return self._plan_running(region)

        # Generate zigzag between edges
        points = []
        min_len = min(len(left), len(right))
        for i in range(min_len):
            # Left side
            x, y = left[i]
            points.append(StitchPoint(
                x=self._px_to_mm(x), y=self._px_to_mm(y),
                stitch_type=StitchType.SATIN,
            ))
            # Right side
            x, y = right[i]
            points.append(StitchPoint(
                x=self._px_to_mm(x), y=self._px_to_mm(y),
                stitch_type=StitchType.SATIN,
            ))

        return points

    def _plan_fill(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate fill stitch (parallel lines across region)."""
        mask = region.mask
        h, w = mask.shape
        if h == 0 or w == 0:
            return []

        # Determine fill angle (default 45°)
        angle = self._compute_fill_angle(region)
        angle_rad = np.radians(angle)

        points = []
        step_px = max(1, int(1.0 / self.stitch_density * self.resolution))

        # Generate parallel scan lines
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        max_diag = int(np.sqrt(h ** 2 + w ** 2))

        for offset in range(-max_diag, max_diag, step_px):
            # Line: x*cos(a) + y*sin(a) = offset
            line_points = []
            for y in range(0, h, step_px):
                x = int((offset - y * sin_a) / cos_a) if abs(cos_a) > 0.01 else 0
                if 0 <= x < w and mask[y, x] == 1:
                    line_points.append((x, y))

            # Add points where line is inside mask
            for x, y in line_points:
                points.append(StitchPoint(
                    x=self._px_to_mm(x), y=self._px_to_mm(y),
                    stitch_type=StitchType.FILL,
                ))

        return points

    def _plan_tatami(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate tatami fill (dense offset rows)."""
        mask = region.mask
        h, w = mask.shape
        if h == 0 or w == 0:
            return []

        step_px = max(1, int(0.8 / self.stitch_density * self.resolution))
        points = []
        row_offset = 0

        for y in range(0, h, step_px):
            # Offset every other row for tatami pattern
            offset = (step_px // 2) * row_offset
            for x in range(offset, w, step_px):
                if mask[y, x] == 1:
                    points.append(StitchPoint(
                        x=self._px_to_mm(x), y=self._px_to_mm(y),
                        stitch_type=StitchType.TATAMI,
                    ))
            row_offset = 1 - row_offset

        return points

    def _plan_french_knot(self, region: ImageRegion) -> List[StitchPoint]:
        """Generate French knot (single point at centroid)."""
        cx, cy = region.centroid
        return [StitchPoint(
            x=self._px_to_mm(cx), y=self._px_to_mm(cy),
            stitch_type=StitchType.FRENCH_KNOT,
        )]

    def _sort_contour(self, contour: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Sort contour points for continuous path (nearest-neighbor)."""
        if len(contour) <= 2:
            return contour

        remaining = list(contour)
        sorted_pts = [remaining.pop(0)]

        while remaining:
            last = sorted_pts[-1]
            # Find nearest remaining point
            min_dist = float('inf')
            min_idx = 0
            for i, pt in enumerate(remaining):
                dist = (pt[0] - last[0]) ** 2 + (pt[1] - last[1]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            sorted_pts.append(remaining.pop(min_idx))

        return sorted_pts

    def _split_contour_pair(self, contour: List[Tuple[int, int]]
                             ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Split contour into left and right edges for satin stitch."""
        if len(contour) < 4:
            return [], []

        # Sort by y, then split at midpoint
        sorted_by_y = sorted(contour, key=lambda p: (p[1], p[0]))
        mid = len(sorted_by_y) // 2

        # Left edge: lower x values
        left = sorted(sorted_by_y[:mid], key=lambda p: p[1])
        # Right edge: higher x values
        right = sorted(sorted_by_y[mid:], key=lambda p: p[1])

        return left, right

    def _compute_fill_angle(self, region: ImageRegion) -> float:
        """Compute optimal fill angle based on region shape."""
        bbox = region.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w > h * 2:
            return 90  # vertical lines for wide regions
        elif h > w * 2:
            return 0   # horizontal lines for tall regions
        else:
            return 45  # diagonal for square-ish regions
