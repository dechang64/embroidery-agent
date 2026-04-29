"""
Image processing module for Embroidery Agent.

Handles:
    - Image preprocessing (resize, normalize)
    - Edge detection (Canny, Sobel)
    - Color segmentation (K-means)
    - Region extraction (contour detection, bounding boxes)
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
    RUNNING = "running"           # 平针 — outlines, details
    SATIN = "satin"               # 缎纹针 — borders, text
    FILL = "fill"                 # 填充针 — solid areas
    CHAIN = "chain"               # 链式针 — decorative outlines
    ZIGZAG = "zigzag"             # 锯齿针 — decorative borders
    CROSS = "cross"               # 十字针 — counted embroidery
    FRENCH_KNOT = "french_knot"   # 法式结 — dots, accents
    TATAMI = "tatami"             # 榻榻米针 — dense fill patterns
    SEED = "seed"                 # 散点针 — scattered texture


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


@dataclass
class ProcessedImage:
    """Result of image processing pipeline."""
    original_size: Tuple[int, int]
    regions: List[ImageRegion]
    color_palette: List[EmbroideryColor]
    edge_map: Optional[np.ndarray] = None
    segment_map: Optional[np.ndarray] = None


class ImageProcessor:
    """Processes input images into embroidery-ready regions."""

    def __init__(self, max_colors: int = 8, min_region_area: int = 100):
        self.max_colors = max_colors
        self.min_region_area = min_region_area

    def process(self, image: Image.Image) -> ProcessedImage:
        """Full processing pipeline: segment → extract regions → assign stitch types."""
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]

        # Color segmentation via K-means
        palette, segment_map = self._kmeans_segment(img_array)

        # Edge detection
        gray = np.array(image.convert("L"))
        edge_map = self._detect_edges(gray)

        # Extract regions from segments
        regions = self._extract_regions(segment_map, palette, (w, h))

        # Assign stitch types
        regions = self._assign_stitch_types(regions, (w, h))

        return ProcessedImage(
            original_size=(w, h),
            regions=regions,
            color_palette=palette,
            edge_map=edge_map,
            segment_map=segment_map,
        )

    def _kmeans_segment(self, img: np.ndarray) -> Tuple[List[EmbroideryColor], np.ndarray]:
        """K-means color segmentation."""
        pixels = img.reshape(-1, 3).astype(np.float32)
        n = len(pixels)

        # Initialize centroids randomly
        rng = np.random.RandomState(42)
        indices = rng.choice(n, min(self.max_colors, n), replace=False)
        centroids = pixels[indices].copy()

        for _ in range(20):
            # Assign pixels to nearest centroid
            dists = np.linalg.norm(pixels[:, None] - centroids[None], axis=2)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.array([pixels[labels == k].mean(axis=0) if (labels == k).any() else centroids[k] for k in range(len(centroids))])
            if np.allclose(centroids, new_centroids, atol=1e-3):
                break
            centroids = new_centroids

        # Build palette
        palette = []
        for c in centroids:
            rgb = tuple(int(max(0, min(255, x))) for x in c)
            name = self._color_name(rgb)
            palette.append(EmbroideryColor(name=name, rgb=rgb))

        segment_map = labels.reshape(img.shape[:2])
        return palette, segment_map

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Sobel edge detection — uses opencv if available, falls back to pure numpy."""
        try:
            import cv2
            gray_u8 = gray.astype(np.uint8)
            sobel_x = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        except ImportError:
            # Pure numpy Sobel fallback
            gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            gy = gx.T
            sx = np.zeros_like(gray, dtype=np.float32)
            sy = np.zeros_like(gray, dtype=np.float32)
            gray_f = gray.astype(np.float32)
            for i in range(1, gray.shape[0] - 1):
                for j in range(1, gray.shape[1] - 1):
                    patch = gray_f[i-1:i+2, j-1:j+2]
                    sx[i, j] = np.sum(patch * gx)
                    sy[i, j] = np.sum(patch * gy)
            magnitude = np.sqrt(sx ** 2 + sy ** 2)
        return (magnitude > np.percentile(magnitude, 85)).astype(np.uint8)

    def _extract_regions(self, segment_map: np.ndarray, palette: List[EmbroideryColor],
                         size: Tuple[int, int]) -> List[ImageRegion]:
        """Extract connected regions from segment map."""
        regions = []
        h, w = segment_map.shape
        visited = np.zeros_like(segment_map, dtype=bool)
        region_id = 0

        for label_idx in range(len(palette)):
            mask = segment_map == label_idx
            if mask.sum() < self.min_region_area:
                continue

            # Simple contour extraction
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = int(mask.sum())
            cx, cy = int(xs.mean()), int(ys.mean())

            # Extract boundary pixels as contour
            contour = self._extract_contour(mask)

            region = ImageRegion(
                region_id=region_id,
                bbox=(x1, y1, x2, y2),
                area=area,
                centroid=(cx, cy),
                color=palette[label_idx],
                contour=contour,
            )
            regions.append(region)
            region_id += 1

        return regions

    def _extract_contour(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Extract boundary pixels from binary mask — uses opencv if available, falls back to pure numpy."""
        try:
            import cv2
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(mask.astype(np.uint8), kernel)
            boundary = mask & ~eroded
        except ImportError:
            # Pure numpy erosion: a pixel survives only if all 9 neighbors are 1
            padded = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
            i_max, j_max = padded.shape
            eroded = np.zeros_like(padded)
            for i in range(1, i_max - 1):
                for j in range(1, j_max - 1):
                    if padded[i, j] == 1 and np.sum(padded[i-1:i+2, j-1:j+2]) == 9:
                        eroded[i, j] = 1
            boundary = mask.astype(bool) & ~eroded[1:-1, 1:-1].astype(bool)
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
