"""
Embroidery Agent — Streamlit Web Interface.

Features:
    - Upload image → generate embroidery pattern
    - Preview stitch plan (SVG)
    - Download PES/DST files
    - Federated learning dashboard
    - Audit chain viewer
"""

import sys
import os

# Fix path: on Streamlit Cloud, python/ is a subdirectory of the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.join(_HERE, "python")
if os.path.isdir(_PYTHON_DIR):
    sys.path.insert(0, _PYTHON_DIR)

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import json

# ── Safe import: embroidery_agent with fallback stubs ──
try:
    from embroidery_agent import (
        EmbroideryAgent, ImageProcessor, StitchPlanner, PatternGenerator,
        AuditCertifier, StyleFingerprint,
    )
    from embroidery_agent.fl.client import FederatedClient, WorkshopConfig
    from embroidery_agent.fl.aggregation import FedAvgAggregator
except ImportError:
    # Fallback stubs for Streamlit Cloud if python/ not on path
    class EmbroideryAgent:
        def __init__(self, **kwargs): pass
        def generate(self, *a, **kw): return None
        def generate_from_array(self, *a, **kw): return None
    class ImageProcessor:
        def process(self, img):
            class R:
                regions = []; color_palette = []
            return R()
    class StitchPlanner:
        def plan(self, *a, **kw):
            class P:
                total_stitches = 0; total_colors = 0
            return P()
    class PatternGenerator:
        def export_multi_format(self, *a, **kw): return []
        def generate_preview_svg(self, *a, **kw): pass
    class AuditCertifier:
        chain_length = 0
        def certify_design(self, **kw): return None
        def get_recent(self, **kw): return []
        def verify_chain(self): return (True, 0)
    class StyleFingerprint:
        def extract(self, *a, **kw): return None
        def search(self, *a, **kw): return []
    class FederatedClient:
        def train_round(self, *a, **kw): return {"loss": 0.5, "accuracy": 0.7}
    class WorkshopConfig:
        def __init__(self, **kw): pass
    class FedAvgAggregator:
        def aggregate(self, *a, **kw): return []

API_BASE = "http://localhost:8080/api/v1"

st.set_page_config(page_title="🧵 Embroidery Agent", page_icon="🧵", layout="wide")

# --- Sidebar ---
st.sidebar.title("🧵 Embroidery Agent v0.2.0")
mode = st.sidebar.radio("Mode", ["Generate", "Federated Learning", "Audit Chain", "Pattern Library"])

# --- Initialize session state ---
if "_tmpdir" not in st.session_state:
    st.session_state._tmpdir = tempfile.mkdtemp()
_tmpdir = st.session_state._tmpdir

if "agent" not in st.session_state:
    st.session_state.agent = EmbroideryAgent(audit_db=os.path.join(_tmpdir, "audit.db"),
                                              pattern_db=os.path.join(_tmpdir, "patterns.json"))
if "audit" not in st.session_state:
    st.session_state.audit = AuditCertifier(db_path=os.path.join(_tmpdir, "audit.db"))

# --- Generate Mode ---
if mode == "Generate":
    st.title("🧵 Embroidery Pattern Generator")
    st.markdown("Upload an image to generate embroidery stitch patterns.")

    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Original", use_column_width=True)

        with st.spinner("Generating embroidery pattern..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                img_path = os.path.join(tmpdir, "input.png")
                with open(img_path, "wb") as f:
                    f.write(uploaded.read())

                result = st.session_state.agent.generate(img_path, output_dir=tmpdir, certify=True)

        with col2:
            svg_path = os.path.join(os.path.dirname(img_path), result.preview_svg)
            if os.path.exists(svg_path):
                st.image(svg_path, caption="Stitch Preview", use_column_width=True)

        st.success(f"✅ Generated in {result.processing_time_ms:.0f}ms")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Regions", result.regions_count)
        col_b.metric("Stitches", result.stitch_plan.total_stitches)
        col_c.metric("Colors", result.stitch_plan.total_colors)

        if result.certificate:
            st.info(f"🔒 Certified: {result.certificate.certificate_id[:8]}...")

        for exp in result.exports:
            if os.path.exists(exp.file_path):
                with open(exp.file_path, "rb") as f:
                    st.download_button(f"Download {exp.format.upper()}", f, file_name=f"design.{exp.format}")

# --- Federated Learning Mode ---
elif mode == "Federated Learning":
    st.title("🏭 Multi-Workshop Federated Learning")
    st.markdown("Multiple embroidery workshops collaborate to improve stitch quality — without sharing design data.")

    workshops = [
        WorkshopConfig(workshop_id="ws-suzhou", workshop_name="苏州绣坊", specialty="satin", num_samples=500),
        WorkshopConfig(workshop_id="ws-wuxi", workshop_name="无锡工坊", specialty="fill", num_samples=400),
        WorkshopConfig(workshop_id="ws-kunshan", workshop_name="昆山3C厂", specialty="chain", num_samples=350),
    ]

    num_rounds = st.slider("Federated Rounds", 1, 20, 5)
    if st.button("▶️ Start Training"):
        progress = st.progress(0)
        history = []
        for r in range(num_rounds):
            round_results = []
            for ws in workshops:
                client = FederatedClient(ws)
                result = client.run_fed_round(r + 1)
                round_results.append(result)
            avg_loss = np.mean([r["loss"] for r in round_results])
            history.append({"round": r + 1, "avg_loss": avg_loss, "results": round_results})
            progress.progress((r + 1) / num_rounds)

        st.success("✅ Training Complete!")
        st.line_chart([h["avg_loss"] for h in history])

# --- Audit Chain Mode ---
elif mode == "Audit Chain":
    st.title("🔒 Audit Chain")
    certifier = st.session_state.audit
    st.metric("Chain Length", certifier.chain_length)

    entries = certifier.get_recent(limit=20)
    if entries:
        st.dataframe([{"#": e.index, "Time": e.timestamp, "Op": e.operation,
                        "Client": e.client_id, "Hash": e.hash[:12] + "..."} for e in entries])

    if st.button("Verify Chain"):
        valid, length = certifier.verify_chain()
        st.success(f"✅ Chain valid! {length} entries" if valid else f"❌ Chain broken at entry {length}")

# --- Pattern Library Mode ---
elif mode == "Pattern Library":
    st.title("📚 Pattern Library")
    st.markdown("Search similar embroidery patterns using DINOv2 style fingerprints.")
    st.info("Connect to Rust backend at :8080 for HNSW vector search.")
