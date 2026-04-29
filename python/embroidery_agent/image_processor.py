"""
Image processing module for Embroidery Agent.

Handles:
    - Image preprocessing (resize, normalize)
    - Color quantization (fast median-cut via PIL)
    - Region extraction (connected components)
    - Stitch type assignment based on region characteristics
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import colorsys


class StitchType(Enum):
    """Embroidery stitch types — 9 categories from traditional Chinese embroidery."""
    RUNNING = "running"
    SATIN = "satin"
    FILL = "fill"
    CHAIN = "chain"
    ZIGZAG = "zigzag"
    CROSS = "cross"
    FRENCH_KNOT = "french_knot"
    TATAMI = "tatami"
    SEED = "seed"


@dataclass
class EmbroideryColor:
    """Thread color with name and RGB values."""
    name: str
    rgb: Tuple[int, int, int]
    thread_code: str = ""

    @property
    def hex(self) -> str:
        return f"#{self.rgb[0]:02x}{self.rgb[1]:02x}{self.rgb[2]:02x}"


@dataclass
class ImageRegion:
    """A detected region in the input image."""
    region_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    centroid: Tuple[int, int]
    color: Optional[EmbroideryColor] = None
    stitch_type: StitchType = StitchType.FILL
    priority: int = 50
    contour: List[Tuple[int, int]] = field(default_factory=list)
    mask: Optional[np.ndarray] = None  # boolean mask (h, w), cropped to bbox


@dataclass
class ProcessedImage:
    """Result of image processing pipeline."""
    original_size: Tuple[int, int]
    regions: List[ImageRegion] = field(default_factory=list)
    color_palette: List[EmbroideryColor] = field(default_factory=list)
    edge_map: Optional[np.ndarray] = None
    segment_map: Optional[np.ndarray] = None
    quantized_image: Optional[Image.Image] = None  # color-quantized PIL Image


class ImageProcessor:
    """Processes images for embroidery pattern generation."""

    def __init__(self, max_colors: int = 8, min_region_area: int = 100):
        self.max_colors = max_colors
        self.min_region_area = min_region_area

    def process(self, image: Image.Image) -> ProcessedImage:
        """Full processing pipeline: quantize → extract regions → assign stitch types."""
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]

        # Fast color quantization using PIL median-cut
        quantized = image.quantize(colors=self.max_colors, method=Image.Quantize.MEDIANCUT)
        segment_map = np.array(quantized, dtype=np.int32)

        # Build palette from quantized image
        raw_palette = quantized.getpalette()  # 768 RGB values (256 * 3)
        # Find actual colors used in the image
        used_indices = np.unique(segment_map).tolist()
        palette = []
        idx_to_palette = {}  # segment_map value → palette index
        for pi, idx in enumerate(used_indices):
            r, g, b = raw_palette[idx*3], raw_palette[idx*3+1], raw_palette[idx*3+2]
            palette.append(EmbroideryColor(name=self._color_name((r, g, b)), rgb=(r, g, b)))
            idx_to_palette[idx] = pi

        # Edge detection
        gray = np.array(image.convert("L"))
        edge_map = self._detect_edges(gray)

        # Extract regions from segments
        regions = self._extract_regions(segment_map, palette, idx_to_palette, (w, h))

        # Assign stitch types
        regions = self._assign_stitch_types(regions, (w, h))

        return ProcessedImage(
            original_size=(w, h),
            regions=regions,
            color_palette=palette,
            edge_map=edge_map,
            segment_map=segment_map,
            quantized_image=quantized.convert("RGB"),
        )

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Sobel edge detection."""
        from scipy.ndimage import convolve
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        sx = convolve(gray.astype(np.float32), gx)
        sy = convolve(gray.astype(np.float32), gy)
        magnitude = np.sqrt(sx ** 2 + sy ** 2)
        return (magnitude > np.percentile(magnitude, 85)).astype(np.uint8)

    def _extract_regions(self, segment_map: np.ndarray, palette: List[EmbroideryColor],
                         idx_to_palette: Dict[int, int],
                         size: Tuple[int, int]) -> List[ImageRegion]:
        """Extract connected regions from segment map using scipy.ndimage.label."""
        from scipy.ndimage import label as ndlabel

        regions = []
        h, w = segment_map.shape
        region_id = 0

        for seg_idx, pal_idx in idx_to_palette.items():
            color_mask = segment_map == seg_idx
            if color_mask.sum() < self.min_region_area:
                continue

            # Skip near-white background (avg brightness > 240)
            rgb = palette[pal_idx].rgb
            if sum(rgb) / 3 > 240:
                continue

            # Split this color into connected components
            labeled, num_components = ndlabel(color_mask)

            for comp_id in range(1, num_components + 1):
                comp_mask = labeled == comp_id
                area = int(comp_mask.sum())
                if area < self.min_region_area:
                    continue

                ys, xs = np.where(comp_mask)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cx, cy = int(xs.mean()), int(ys.mean())

                # Crop mask to bbox for efficient storage
                cropped_mask = comp_mask[y1:y2 + 1, x1:x2 + 1].copy()

                # Extract boundary pixels as contour
                contour = self._extract_contour(comp_mask)

                region = ImageRegion(
                    region_id=region_id,
                    bbox=(x1, y1, x2, y2),
                    area=area,
                    centroid=(cx, cy),
                    color=palette[pal_idx],
                    contour=contour,
                    mask=cropped_mask,
                )
                regions.append(region)
                region_id += 1

        return regions

    def _extract_contour(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Extract boundary pixels from binary mask."""
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        ys, xs = np.where(boundary)
        return list(zip(xs.tolist(), ys.tolist()))

    def _assign_stitch_types(self, regions: List[ImageRegion],
                             size: Tuple[int, int]) -> List[ImageRegion]:
        """Assign stitch types based on region characteristics."""
        h, w = size
        total_area = h * w

        for region in regions:
            area_ratio = region.area / total_area
            bbox_w = region.bbox[2] - region.bbox[0]
            bbox_h = region.bbox[3] - region.bbox[1]
            aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

            if area_ratio < 0.005:
                region.stitch_type = StitchType.FRENCH_KNOT
                region.priority = 90
            elif aspect_ratio > 5:
                region.stitch_type = StitchType.SATIN
                region.priority = 50
            elif area_ratio > 0.15:
                region.stitch_type = StitchType.TATAMI
                region.priority = 10
            else:
                region.stitch_type = StitchType.FILL
                region.priority = 30

        return regions

    @staticmethod
    def _color_name(rgb: Tuple[int, int, int]) -> str:
        """Generate a human-readable color name."""
        h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        if s < 0.1:
            return "gray" if v > 0.5 else "dark"
        if v < 0.2:
            return "black"
        hue_names = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]
        idx = int(h * 8) % 8
        return hue_names[idx]
