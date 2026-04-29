"""
Embroidery Agent — Streamlit Web Interface.

Features:
    - Generate: Upload image → embroidery pattern preview + download
    - Pattern Library: Browse generated patterns, search by style
    - Federated Learning: Multi-workshop training simulation
    - Audit Chain: Blockchain-style design certification log
"""

import sys
import os
from io import BytesIO
from pathlib import Path

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
    _REAL_AGENT = True
except ImportError:
    _REAL_AGENT = False
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
        def generate_preview_png(self, *a, **kw): pass
    class AuditCertifier:
        chain_length = 0
        def certify_design(self, **kw): return None
        def get_recent(self, **kw): return []
        def verify_chain(self): return (True, 0)
    class StyleFingerprint:
        def compute(self, *a, **kw): return np.zeros(768, dtype=np.float32)
        def search(self, *a, **kw): return []
    class FederatedClient:
        def run_fed_round(self, *a, **kw): return {"loss": 0.5, "accuracy": 0.7}
    class WorkshopConfig:
        def __init__(self, **kw): pass
    class FedAvgAggregator:
        def aggregate(self, *a, **kw): return []


st.set_page_config(page_title="🧵 Embroidery Agent", page_icon="🧵", layout="wide")

# --- Sidebar ---
st.sidebar.title("🧵 Embroidery Agent v0.2.0")
mode = st.sidebar.radio("Mode", ["🧵 Generate", "📚 Pattern Library", "🏭 Federated Learning", "🔒 Audit Chain"])

# --- Initialize session state ---
if "_tmpdir" not in st.session_state:
    st.session_state._tmpdir = tempfile.mkdtemp()
_tmpdir = st.session_state._tmpdir

if "agent" not in st.session_state:
    st.session_state.agent = EmbroideryAgent(
        audit_db=os.path.join(_tmpdir, "audit.db"),
        pattern_db=os.path.join(_tmpdir, "patterns.json"),
    )
if "audit" not in st.session_state:
    st.session_state.audit = AuditCertifier(db_path=os.path.join(_tmpdir, "audit.db"))

# Pattern library stored in session
if "pattern_library" not in st.session_state:
    st.session_state.pattern_library = []  # list of {name, preview_path, timestamp, colors, stitches}


# ═══════════════════════════════════════════════════════════════
# --- 🧵 Generate Mode ---
# ═══════════════════════════════════════════════════════════════
if mode == "🧵 Generate":
    st.title("🧵 Embroidery Pattern Generator")
    st.markdown("Upload an image to generate an embroidery stitch pattern. "
                "The system will segment colors, plan stitches, and produce a preview.")

    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Original", use_container_width=True)

        with st.spinner("Generating embroidery pattern..."):
            tmpdir = tempfile.mkdtemp()
            img_path = os.path.join(tmpdir, "input.png")
            with open(img_path, "wb") as f:
                f.write(uploaded.read())

            try:
                result = st.session_state.agent.generate(img_path, output_dir=tmpdir, certify=True)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                result = None

        if result is None:
            st.warning("Generation failed. The core library may not be available.")
        else:
            # Show preview
            with col2:
                preview_png = getattr(result, 'preview_png', '')
                preview_svg = result.preview_svg
                if preview_png and os.path.exists(preview_png):
                    st.image(preview_png, caption="Stitch Preview", use_container_width=True)
                elif preview_svg and os.path.exists(preview_svg):
                    svg_content = Path(preview_svg).read_text(encoding="utf-8")
                    st.markdown(
                        f'<div style="text-align:center">{svg_content}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("Stitch Preview")
                else:
                    st.info("Preview not available")

            st.success(f"✅ Generated in {result.processing_time_ms:.0f}ms")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Regions", result.regions_count)
            col_b.metric("Stitches", result.stitch_plan.total_stitches)
            col_c.metric("Colors", result.stitch_plan.total_colors)

            if result.certificate:
                st.info(f"🔒 Certified: {result.certificate.certificate_id[:8]}...")

            # Download buttons
            if result.exports:
                dl_cols = st.columns(len(result.exports))
                for i, exp in enumerate(result.exports):
                    if os.path.exists(exp.file_path):
                        with open(exp.file_path, "rb") as f:
                            file_data = f.read()
                        dl_cols[i].download_button(
                            f"⬇️ {exp.format.upper()} ({exp.file_size_bytes:,} bytes)",
                            data=file_data,
                            file_name=f"design.{exp.format}",
                            key=f"dl_{exp.format}_{i}",
                        )
                    else:
                        dl_cols[i].warning(f"{exp.format.upper()} file not found")
            else:
                st.warning("No export files generated.")

            # Save to pattern library
            if st.button("💾 Save to Pattern Library"):
                entry = {
                    "name": uploaded.name,
                    "preview_path": preview_png if preview_png and os.path.exists(preview_png) else "",
                    "timestamp": st.session_state._tmpdir,
                    "colors": result.stitch_plan.total_colors,
                    "stitches": result.stitch_plan.total_stitches,
                    "regions": result.regions_count,
                }
                st.session_state.pattern_library.append(entry)
                st.success(f"Saved! Library now has {len(st.session_state.pattern_library)} patterns.")


# ═══════════════════════════════════════════════════════════════
# --- 📚 Pattern Library Mode ---
# ═══════════════════════════════════════════════════════════════
elif mode == "📚 Pattern Library":
    st.title("📚 Pattern Library")
    st.markdown("Browse your generated embroidery patterns. Patterns are saved from the Generate tab.")

    library = st.session_state.pattern_library

    if not library:
        st.info("No patterns saved yet. Go to **Generate** tab, create a pattern, and click **Save to Pattern Library**.")
    else:
        st.metric("Total Patterns", len(library))

        # Display patterns in a grid
        cols = st.columns(min(len(library), 4))
        for i, entry in enumerate(library):
            with cols[i % len(cols)]:
                st.caption(f"**{entry['name']}**")
                if entry.get("preview_path") and os.path.exists(entry["preview_path"]):
                    st.image(entry["preview_path"], use_container_width=True)
                else:
                    st.warning("Preview unavailable")
                c1, c2 = st.columns(2)
                c1.metric("Colors", entry.get("colors", 0))
                c2.metric("Stitches", entry.get("stitches", 0))

        # Clear library
        if st.button("🗑️ Clear Library"):
            st.session_state.pattern_library = []
            st.rerun()

    # --- Style Search (standalone, no Rust backend needed) ---
    st.divider()
    st.subheader("🔍 Style Search")
    st.markdown("Upload a reference image to find similar patterns by color distribution.")

    query_img = st.file_uploader("Upload reference image", type=["png", "jpg", "jpeg"],
                                  key="style_search")
    if query_img and _REAL_AGENT:
        col_q, col_r = st.columns(2)
        with col_q:
            st.image(query_img, caption="Reference", use_container_width=True)
        with col_r:
            with st.spinner("Computing style fingerprint..."):
                try:
                    tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp_f.write(query_img.read())
                    tmp_f.close()
                    ref_img = Image.open(tmp_f.name).convert("RGB")
                    fingerprint = StyleFingerprint()
                    feature = fingerprint.compute(ref_img)
                    st.success(f"✅ Fingerprint computed ({len(feature)}-dim vector)")
                    st.json({
                        "dimension": len(feature),
                        "norm": float(np.linalg.norm(feature)),
                        "mean": float(feature.mean()),
                        "std": float(feature.std()),
                    })
                    os.unlink(tmp_f.name)
                except Exception as e:
                    st.error(f"Fingerprint failed: {e}")
    elif query_img and not _REAL_AGENT:
        st.warning("Style search requires the embroidery_agent library.")


# ═══════════════════════════════════════════════════════════════
# --- 🏭 Federated Learning Mode ---
# ═══════════════════════════════════════════════════════════════
elif mode == "🏭 Federated Learning":
    st.title("🏭 Multi-Workshop Federated Learning")
    st.markdown(
        "Simulate federated learning across multiple embroidery workshops. "
        "Each workshop trains locally on its proprietary stitch data, "
        "then submits weight updates to a central server for aggregation (FedAvg). "
        "**No raw design data is shared between workshops.**"
    )

    st.subheader("Workshop Configuration")
    col_ws1, col_ws2, col_ws3 = st.columns(3)

    with col_ws1:
        st.markdown("#### 🏺 苏州绣坊")
        st.caption("Specialty: Satin stitch")
        st.metric("Samples", 500)

    with col_ws2:
        st.markdown("#### 🏺 无锡工坊")
        st.caption("Specialty: Fill stitch")
        st.metric("Samples", 400)

    with col_ws3:
        st.markdown("#### 🏺 昆山3C厂")
        st.caption("Specialty: Chain stitch")
        st.metric("Samples", 350)

    st.divider()

    num_rounds = st.slider("Federated Rounds", 1, 20, 5)
    st.caption("Each round: all workshops train locally → submit updates → server aggregates")

    if st.button("▶️ Start Training"):
        workshops = [
            WorkshopConfig(workshop_id="ws-suzhou", workshop_name="苏州绣坊",
                           specialty="satin", num_samples=500),
            WorkshopConfig(workshop_id="ws-wuxi", workshop_name="无锡工坊",
                           specialty="fill", num_samples=400),
            WorkshopConfig(workshop_id="ws-kunshan", workshop_name="昆山3C厂",
                           specialty="chain", num_samples=350),
        ]

        progress = st.progress(0)
        history = []
        round_details = []

        for r in range(num_rounds):
            round_results = []
            for ws in workshops:
                client = FederatedClient(ws)
                result = client.run_fed_round(r + 1)
                round_results.append(result)

            avg_loss = np.mean([res["loss"] for res in round_results])
            accepted = sum(1 for res in round_results if res.get("accepted", False))
            history.append({"round": r + 1, "avg_loss": avg_loss, "accepted": accepted})
            round_details.append({
                "round": r + 1,
                **{f"{res['workshop_id']}_loss": res["loss"] for res in round_results}
            })
            progress.progress((r + 1) / num_rounds)

        st.success("✅ Training Complete!")

        # Results
        col_loss, col_acc = st.columns(2)
        with col_loss:
            st.subheader("Loss Curve")
            st.line_chart([h["avg_loss"] for h in history])
            st.caption("Average loss across all workshops per round")

        with col_acc:
            st.subheader("Acceptance Rate")
            st.bar_chart([h["accepted"] for h in history])
            st.caption("Number of workshop updates accepted per round")

        st.subheader("Per-Round Details")
        st.dataframe(round_details, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# --- 🔒 Audit Chain Mode ---
# ═══════════════════════════════════════════════════════════════
elif mode == "🔒 Audit Chain":
    st.title("🔒 Audit Chain")
    st.markdown(
        "Blockchain-style audit trail for embroidery design certification. "
        "Each generated pattern is cryptographically hashed and chained — "
        "tampering with any entry breaks the chain."
    )

    certifier = st.session_state.audit
    st.metric("Chain Length", certifier.chain_length)

    entries = certifier.get_recent(limit=20)
    if entries:
        st.subheader("Recent Entries")
        st.dataframe(
            [{"#": e.index, "Time": e.timestamp, "Operation": e.operation,
              "Client": e.client_id, "Hash": e.hash[:16] + "...",
              "Prev Hash": e.prev_hash[:16] + "..."}
             for e in entries],
            use_container_width=True,
        )
    else:
        st.info("No audit entries yet. Generate a pattern in the **Generate** tab to create entries.")

    col_v, col_s = st.columns(2)
    with col_v:
        if st.button("🔍 Verify Chain Integrity"):
            valid, length = certifier.verify_chain()
            if valid:
                st.success(f"✅ Chain valid! {length} entries verified.")
            else:
                st.error(f"❌ Chain broken at entry {length}!")

    with col_s:
        st.markdown("### How it works")
        st.markdown("""
        1. Each pattern generation creates an audit entry
        2. Entry hash = SHA256(index + timestamp + operation + data + prev_hash)
        3. Each entry links to the previous via `prev_hash`
        4. Verifying the chain ensures no entry was tampered with
        """)
