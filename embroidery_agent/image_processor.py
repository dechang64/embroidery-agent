"""
Image processing module for Embroidery Agent.

Handles:
    - Image preprocessing (resize, normalize)
    - Edge detection (Canny, Sobel)
    - Color segmentation (K-means, region growing)
    - Region extraction (contour detection, bounding boxes)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import colorsys


class StitchType(Enum):
    """Embroidery stitch types."""
    RUNNING = "running"       # 平针 — outlines, details
    SATIN = "satin"           # 缎纹针 — borders, text
    FILL = "fill"             # 填充针 — solid areas
    CHAIN = "chain"           # 链式针 — decorative outlines
    ZIGZAG = "zigzag"         # 锯齿针 — decorative borders
    CROSS = "cross"           # 十字针 — counted embroidery
    FRENCH_KNOT = "french_knot"  # 法式结 — dots, accents
    TATAMI = "tatami"         # 榻榻米针 — dense fill patterns


@dataclass
class EmbroideryColor:
    """Thread color with name and RGB values."""
    name: str
    rgb: Tuple[int, int, int]
    thread_code: str = ""     # e.g., "Brother_001"

    @property
    def hex(self) -> str:
        return f"#{self.rgb[0]:02x}{self.rgb[1]:02x}{self.rgb[2]:02x}"


@dataclass
class ImageRegion:
    """A segmented region of the input image."""
    region_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray                  # binary mask
    dominant_color: Tuple[int, int, int]
    area: int = 0
    centroid: Tuple[float, float] = (0.0, 0.0)
    contour: List[Tuple[int, int]] = field(default_factory=list)
    stitch_type: StitchType = StitchType.FILL
    priority: int = 0  # lower = stitch first


@dataclass
class ProcessedImage:
    """Result of image processing pipeline."""
    original_size: Tuple[int, int]
    regions: List[ImageRegion] = field(default_factory=list)
    edges: Optional[np.ndarray] = None
    color_palette: List[EmbroideryColor] = field(default_factory=list)
    background_color: Optional[Tuple[int, int, int]] = None


# Standard embroidery thread palette (Madeira Rayon 40)
THREAD_PALETTE = [
    EmbroideryColor("Black", (0, 0, 0), "Madeira_0001"),
    EmbroideryColor("White", (255, 255, 255), "Madeira_1000"),
    EmbroideryColor("Red", (220, 20, 20), "Madeira_1012"),
    EmbroideryColor("Dark Red", (139, 0, 0), "Madeira_1020"),
    EmbroideryColor("Royal Blue", (0, 35, 149), "Madeira_1086"),
    EmbroideryColor("Navy", (0, 0, 128), "Madeira_1090"),
    EmbroideryColor("Forest Green", (34, 139, 34), "Madeira_1102"),
    EmbroideryColor("Dark Green", (0, 100, 0), "Madeira_1110"),
    EmbroideryColor("Gold", (255, 215, 0), "Madeira_1072"),
    EmbroideryColor("Orange", (255, 140, 0), "Madeira_1060"),
    EmbroideryColor("Pink", (255, 105, 180), "Madeira_1040"),
    EmbroideryColor("Purple", (128, 0, 128), "Madeira_1130"),
    EmbroideryColor("Brown", (139, 69, 19), "Madeira_1140"),
    EmbroideryColor("Gray", (128, 128, 128), "Madeira_1150"),
    EmbroideryColor("Light Blue", (135, 206, 235), "Madeira_1080"),
    EmbroideryColor("Beige", (245, 222, 179), "Madeira_1160"),
    EmbroideryColor("Coral", (255, 127, 80), "Madeira_1050"),
    EmbroideryColor("Teal", (0, 128, 128), "Madeira_1120"),
    EmbroideryColor("Lavender", (150, 123, 182), "Madeira_1135"),
    EmbroideryColor("Cream", (255, 253, 208), "Madeira_1165"),
]


def find_nearest_thread(rgb: Tuple[int, int, int]) -> EmbroideryColor:
    """Find the nearest thread color from the standard palette."""
    min_dist = float('inf')
    nearest = THREAD_PALETTE[0]
    for thread in THREAD_PALETTE:
        dist = sum((a - b) ** 2 for a, b in zip(rgb, thread.rgb))
        if dist < min_dist:
            min_dist = dist
            nearest = thread
    return nearest


class ImageProcessor:
    """Processes input images into embroidery-ready regions."""

    def __init__(self, max_colors: int = 8, min_region_area: int = 100,
                 resize_max: int = 500):
        self.max_colors = max_colors
        self.min_region_area = min_region_area
        self.resize_max = resize_max

    def process(self, image: Image.Image) -> ProcessedImage:
        """Full processing pipeline: resize → segment → extract regions."""
        original_size = image.size

        # Resize if needed
        if max(image.size) > self.resize_max:
            ratio = self.resize_max / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Edge detection
        edges = self._detect_edges(image)

        # Color segmentation
        regions, palette = self._segment_colors(image)

        # Edge-based regions (outlines)
        edge_regions = self._extract_edge_regions(edges, image.size)

        # Merge and assign stitch types
        all_regions = regions + edge_regions
        all_regions = self._assign_stitch_types(all_regions, image.size)

        # Sort by priority (fill first, then satin, then running)
        all_regions.sort(key=lambda r: r.priority)

        return ProcessedImage(
            original_size=original_size,
            regions=all_regions,
            edges=edges,
            color_palette=palette,
        )

    def _detect_edges(self, image: Image.Image) -> np.ndarray:
        """Detect edges using Canny-like approach with PIL."""
        gray = image.convert("L")
        # Sobel-like edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        # Threshold
        edges = edges.point(lambda x: 255 if x > 50 else 0)
        return np.array(edges)

    def _segment_colors(self, image: Image.Image) -> Tuple[List[ImageRegion], List[EmbroideryColor]]:
        """Segment image by dominant colors using K-means."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # K-means color quantization
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        centers = self._kmeans_colors(pixels, self.max_colors)

        # Map each pixel to nearest center
        distances = np.sqrt(np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1).reshape(h, w)

        # Build thread palette
        palette = []
        for center in centers:
            rgb = tuple(int(c) for c in center)
            thread = find_nearest_thread(rgb)
            palette.append(thread)

        # Extract regions
        regions = []
        for color_idx in range(len(centers)):
            mask = (labels == color_idx).astype(np.uint8)
            area = int(mask.sum())

            if area < self.min_region_area:
                continue

            # Bounding box
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            # Centroid
            centroid = (float(xs.mean()), float(ys.mean()))

            # Simple contour (boundary pixels)
            contour = self._extract_contour(mask)

            region = ImageRegion(
                region_id=color_idx,
                bbox=bbox,
                mask=mask,
                dominant_color=palette[color_idx].rgb,
                area=area,
                centroid=centroid,
                contour=contour,
            )
            regions.append(region)

        return regions, palette

    def _kmeans_colors(self, pixels: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """Simple K-means clustering for color quantization."""
        n = len(pixels)
        # Initialize centers using evenly spaced samples
        indices = np.linspace(0, n - 1, k, dtype=int)
        centers = pixels[indices].copy()

        for _ in range(max_iter):
            # Assign
            distances = np.sqrt(np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2))
            labels = np.argmin(distances, axis=1)

            # Update
            new_centers = np.zeros_like(centers)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    new_centers[i] = pixels[mask].mean(axis=0)
                else:
                    new_centers[i] = centers[i]

            if np.allclose(centers, new_centers, atol=1.0):
                break
            centers = new_centers

        return centers

    def _extract_contour(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Extract contour points from binary mask."""
        h, w = mask.shape
        contour = []
        # Simple boundary detection: pixels with at least one background neighbor
        padded = np.pad(mask, 1, mode='constant', constant_values=0)
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 1:
                    py, px = y + 1, x + 1
                    neighbors = [
                        padded[py-1, px], padded[py+1, px],
                        padded[py, px-1], padded[py, px+1],
                    ]
                    if 0 in neighbors:
                        contour.append((x, y))
        return contour

    def _extract_edge_regions(self, edges: np.ndarray,
                               size: Tuple[int, int]) -> List[ImageRegion]:
        """Convert edge map to outline regions for running stitch."""
        h, w = size
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return []

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        mask = (edges > 0).astype(np.uint8)
        contour = self._extract_contour(mask)

        return [ImageRegion(
            region_id=-1,  # special ID for edges
            bbox=bbox,
            mask=mask,
            dominant_color=(0, 0, 0),
            area=int(mask.sum()),
            centroid=(float(xs.mean()), float(ys.mean())),
            contour=contour,
            stitch_type=StitchType.RUNNING,
            priority=100,  # stitch last (outlines on top)
        )]

    def _assign_stitch_types(self, regions: List[ImageRegion],
                              size: Tuple[int, int]) -> List[ImageRegion]:
        """Assign appropriate stitch types based on region characteristics."""
        h, w = size
        total_area = h * w

        for region in regions:
            if region.stitch_type == StitchType.RUNNING:
                # Already assigned (edge region)
                continue

            area_ratio = region.area / total_area
            bbox_w = region.bbox[2] - region.bbox[0]
            bbox_h = region.bbox[3] - region.bbox[1]
            aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

            if area_ratio < 0.005:
                # Very small region → French knot or running
                region.stitch_type = StitchType.FRENCH_KNOT
                region.priority = 90
            elif aspect_ratio > 5:
                # Long thin region → satin stitch
                region.stitch_type = StitchType.SATIN
                region.priority = 50
            elif area_ratio > 0.15:
                # Large area → tatami fill
                region.stitch_type = StitchType.TATAMI
                region.priority = 10
            else:
                # Medium area → standard fill
                region.stitch_type = StitchType.FILL
                region.priority = 30

        return regions
