"""Tests for Embroidery Agent MVP."""

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile
import os

from embroidery_agent.image_processor import (
    ImageProcessor, ProcessedImage, ImageRegion, StitchType, EmbroideryColor,
)
from embroidery_agent.stitch_planner import (
    StitchPlanner, StitchPlan, StitchBlock, StitchPoint,
)
from embroidery_agent.pattern_generator import PatternGenerator, ExportResult
from embroidery_agent.agent import EmbroideryAgent, GenerationResult


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def simple_image():
    """Create a simple test image: red circle on white background."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(255, 0, 0))
    return img


@pytest.fixture
def multi_color_image():
    """Create a multi-color test image."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, 45, 45], fill=(255, 0, 0))      # red square
    draw.rectangle([55, 5, 95, 45], fill=(0, 255, 0))      # green square
    draw.rectangle([5, 55, 45, 95], fill=(0, 0, 255))      # blue square
    draw.rectangle([55, 55, 95, 95], fill=(255, 255, 0))   # yellow square
    return img


@pytest.fixture
def thin_line_image():
    """Create an image with thin lines (for satin/running stitch)."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(10, 50), (90, 50)], fill=(0, 0, 0), width=2)
    draw.line([(50, 10), (50, 90)], fill=(0, 0, 0), width=2)
    return img


@pytest.fixture
def processor():
    return ImageProcessor(max_colors=4)


@pytest.fixture
def planner():
    return StitchPlanner(stitch_density=3.0, resolution=5.0)


@pytest.fixture
def generator():
    return PatternGenerator()


@pytest.fixture
def agent():
    return EmbroideryAgent(max_colors=4, export_formats=["pes", "dst", "svg"])


@pytest.fixture
def sample_palette():
    return [
        EmbroideryColor("Red", (255, 0, 0), "Brother_001"),
        EmbroideryColor("Green", (0, 255, 0), "Brother_002"),
        EmbroideryColor("Blue", (0, 0, 255), "Brother_003"),
        EmbroideryColor("Black", (0, 0, 0), "Brother_004"),
    ]


@pytest.fixture
def sample_region():
    """Create a sample region (a filled rectangle)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    return ImageRegion(
        region_id=1,
        bbox=(10, 10, 40, 40),
        mask=mask,
        dominant_color=(255, 0, 0),
        area=900,
        centroid=(25, 25),
        contour=[(10, 10), (40, 10), (40, 40), (10, 40)],
        stitch_type=StitchType.FILL,
        priority=30,
    )


@pytest.fixture
def edge_region():
    """Create a thin edge region for running stitch."""
    return ImageRegion(
        region_id=0,
        bbox=(0, 0, 100, 100),
        mask=np.zeros((100, 100), dtype=np.uint8),
        dominant_color=(0, 0, 0),
        area=500,
        centroid=(50, 50),
        contour=[(10, 50), (20, 50), (30, 50), (40, 50), (50, 50),
                 (60, 50), (70, 50), (80, 50), (90, 50)],
        stitch_type=StitchType.RUNNING,
        priority=100,
    )


# ============================================================
# ImageProcessor Tests
# ============================================================

class TestImageProcessor:
    def test_process_simple_image(self, processor, simple_image):
        result = processor.process(simple_image)
        assert isinstance(result, ProcessedImage)
        assert len(result.regions) > 0
        assert len(result.color_palette) > 0

    def test_process_multi_color(self, processor, multi_color_image):
        result = processor.process(multi_color_image)
        assert isinstance(result, ProcessedImage)
        assert len(result.color_palette) >= 2

    def test_process_returns_correct_size(self, processor, simple_image):
        result = processor.process(simple_image)
        assert result.original_size == (100, 100)

    def test_regions_have_required_fields(self, processor, simple_image):
        result = processor.process(simple_image)
        for region in result.regions:
            assert len(region.bbox) == 4
            assert region.mask is not None
            assert len(region.dominant_color) == 3
            assert region.area > 0
            assert len(region.centroid) == 2

    def test_edge_detection(self, processor, simple_image):
        edges = processor._detect_edges(simple_image)
        assert isinstance(edges, np.ndarray)
        assert edges.shape == (100, 100)

    def test_color_segmentation(self, processor, multi_color_image):
        regions, colors = processor._segment_colors(multi_color_image)
        assert isinstance(regions, list)
        assert isinstance(colors, list)
        assert len(colors) >= 2

    def test_palette_colors_are_valid(self, processor, simple_image):
        result = processor.process(simple_image)
        for color in result.color_palette:
            assert all(0 <= c <= 255 for c in color.rgb)
            assert color.name


# ============================================================
# StitchPlanner Tests
# ============================================================

class TestStitchPlanner:
    def test_plan_single_region(self, planner, sample_region, sample_palette):
        plan = planner.plan([sample_region], sample_palette)
        assert isinstance(plan, StitchPlan)
        assert plan.total_stitches > 0
        assert plan.total_colors >= 1

    def test_plan_multiple_regions(self, planner, sample_region, edge_region, sample_palette):
        plan = planner.plan([sample_region, edge_region], sample_palette)
        assert plan.total_stitches > 0
        assert len(plan.blocks) >= 1

    def test_plan_empty_regions(self, planner, sample_palette):
        plan = planner.plan([], sample_palette)
        assert plan.total_stitches == 0

    def test_fill_stitch_generates_points(self, planner, sample_region, sample_palette):
        plan = planner.plan([sample_region], sample_palette)
        assert plan.total_stitches > 0
        # Fill should generate many points
        assert plan.total_stitches > 10

    def test_running_stitch_follows_contour(self, planner, edge_region, sample_palette):
        plan = planner.plan([edge_region], sample_palette)
        assert plan.total_stitches > 0

    def test_plan_design_dimensions(self, planner, sample_region, sample_palette):
        plan = planner.plan([sample_region], sample_palette)
        assert plan.design_width_mm > 0
        assert plan.design_height_mm > 0

    def test_get_all_points(self, planner, sample_region, sample_palette):
        plan = planner.plan([sample_region], sample_palette)
        all_points = plan.get_all_points()
        assert len(all_points) == plan.total_stitches

    def test_stitch_points_have_mm_coords(self, planner, sample_region, sample_palette):
        plan = planner.plan([sample_region], sample_palette)
        for point in plan.get_all_points():
            assert point.x >= 0
            assert point.y >= 0


# ============================================================
# PatternGenerator Tests
# ============================================================

class TestPatternGenerator:
    def _make_plan(self, planner, sample_region, sample_palette):
        return planner.plan([sample_region], sample_palette)

    def test_export_pes(self, generator, planner, sample_region, sample_palette, tmp_path):
        plan = self._make_plan(planner, sample_region, sample_palette)
        result = generator.export(plan, str(tmp_path / "test"), "pes")
        assert isinstance(result, ExportResult)
        assert result.format == "PES"
        assert Path(result.file_path).exists()
        assert result.file_size_bytes > 0

    def test_export_dst(self, generator, planner, sample_region, sample_palette, tmp_path):
        plan = self._make_plan(planner, sample_region, sample_palette)
        result = generator.export(plan, str(tmp_path / "test"), "dst")
        assert isinstance(result, ExportResult)
        assert result.format == "DST"
        assert Path(result.file_path).exists()

    def test_export_svg(self, generator, planner, sample_region, sample_palette, tmp_path):
        plan = self._make_plan(planner, sample_region, sample_palette)
        result = generator.export(plan, str(tmp_path / "test"), "svg")
        assert isinstance(result, ExportResult)
        assert result.format == "SVG"
        assert Path(result.file_path).exists()

    def test_export_multi_format(self, generator, planner, sample_region, sample_palette, tmp_path):
        plan = self._make_plan(planner, sample_region, sample_palette)
        results = generator.export_multi_format(plan, str(tmp_path / "test"),
                                                  ["pes", "dst", "svg"])
        assert len(results) == 3
        for r in results:
            assert Path(r.file_path).exists()

    def test_export_result_metadata(self, generator, planner, sample_region, sample_palette, tmp_path):
        plan = self._make_plan(planner, sample_region, sample_palette)
        result = generator.export(plan, str(tmp_path / "test"), "pes")
        assert result.stitch_count == plan.total_stitches
        assert result.color_count >= 1


# ============================================================
# EmbroideryAgent (End-to-End) Tests
# ============================================================

class TestEmbroideryAgent:
    def test_generate_simple(self, agent, simple_image, tmp_path):
        # Save test image
        img_path = str(tmp_path / "test.png")
        simple_image.save(img_path)

        result = agent.generate(img_path, str(tmp_path), "test")
        assert isinstance(result, GenerationResult)
        assert result.regions_count > 0
        assert result.stitch_plan.total_stitches > 0
        assert result.processing_time_ms > 0

    def test_generate_creates_files(self, agent, simple_image, tmp_path):
        img_path = str(tmp_path / "test.png")
        simple_image.save(img_path)

        result = agent.generate(img_path, str(tmp_path), "test")
        assert len(result.exports) >= 2  # at least pes + dst
        for exp in result.exports:
            assert Path(exp.file_path).exists()

    def test_generate_creates_preview(self, agent, simple_image, tmp_path):
        img_path = str(tmp_path / "test.png")
        simple_image.save(img_path)

        result = agent.generate(img_path, str(tmp_path), "test")
        assert Path(result.preview_svg).exists()

    def test_generate_summary(self, agent, simple_image, tmp_path):
        img_path = str(tmp_path / "test.png")
        simple_image.save(img_path)

        result = agent.generate(img_path, str(tmp_path), "test")
        summary = result.summary
        assert "Stitches:" in summary
        assert "Colors:" in summary

    def test_generate_from_array(self, agent, simple_image, tmp_path):
        arr = np.array(simple_image)
        result = agent.generate_from_array(arr, str(tmp_path), "array_test")
        assert isinstance(result, GenerationResult)
        assert result.stitch_plan.total_stitches > 0

    def test_generate_multi_color(self, agent, multi_color_image, tmp_path):
        img_path = str(tmp_path / "multi.png")
        multi_color_image.save(img_path)

        result = agent.generate(img_path, str(tmp_path), "multi")
        assert result.stitch_plan.total_colors >= 2


# ============================================================
# Data Model Tests
# ============================================================

class TestDataModels:
    def test_embroidery_color(self):
        color = EmbroideryColor("Red", (255, 0, 0), "Brother_001")
        assert color.hex == "#ff0000"
        assert color.name == "Red"

    def test_stitch_point(self):
        point = StitchPoint(x=10.0, y=20.0)
        assert point.as_tuple == (10.0, 20.0)
        assert not point.jump
        assert not point.trim

    def test_stitch_block_bounding_box(self):
        block = StitchBlock(
            color=EmbroideryColor("Red", (255, 0, 0)),
            stitch_type=StitchType.FILL,
            points=[
                StitchPoint(0, 0), StitchPoint(10, 0),
                StitchPoint(10, 10), StitchPoint(0, 10),
            ],
            stitch_count=4,
        )
        bbox = block.bounding_box
        assert bbox == (0, 0, 10, 10)

    def test_stitch_block_empty(self):
        block = StitchBlock(
            color=EmbroideryColor("Red", (255, 0, 0)),
            stitch_type=StitchType.FILL,
        )
        assert block.bounding_box == (0, 0, 0, 0)
        assert block.stitch_count == 0
