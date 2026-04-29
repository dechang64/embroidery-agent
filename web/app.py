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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import json

from embroidery_agent import (
    EmbroideryAgent, ImageProcessor, StitchPlanner, PatternGenerator,
    AuditCertifier, StyleFingerprint,
)
from embroidery_agent.fl.client import FederatedClient, WorkshopConfig
from embroidery_agent.fl.aggregation import FedAvgAggregator

API_BASE = "http://localhost:8080/api/v1"

st.set_page_config(page_title="🧵 Embroidery Agent", page_icon="🧵", layout="wide")

# --- Sidebar ---
st.sidebar.title("🧵 Embroidery Agent v0.2.0")
mode = st.sidebar.radio("Mode", ["Generate", "Federated Learning", "Audit Chain", "Pattern Library"])

# --- Initialize session state ---
if "agent" not in st.session_state:
    with tempfile.TemporaryDirectory() as tmpdir:
        st.session_state.agent = EmbroideryAgent(audit_db=os.path.join(tmpdir, "audit.db"),
                                                  pattern_db=os.path.join(tmpdir, "patterns.json"))
if "audit" not in st.session_state:
    with tempfile.TemporaryDirectory() as tmpdir:
        st.session_state.audit = AuditCertifier(db_path=os.path.join(tmpdir, "audit.db"))

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
