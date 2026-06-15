from __future__ import annotations
"""
Embroidery Agent — 刺绣针迹自动生成系统
Streamlit Cloud App

Features:
    - Upload design image
    - AI color segmentation & region detection
    - Stitch type auto-assignment
    - PES/DST/SVG file generation & download
    - Style fingerprint matching (DINOv2)
    - Audit chain certification
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import time
import json
import os
import base64

from embroidery_agent.image_processor import ImageProcessor, StitchType
from embroidery_agent.stitch_planner import StitchPlanner
from embroidery_agent.pattern_generator import PatternGenerator
from embroidery_agent.agent import EmbroideryAgent
from embroidery_agent.audit_certifier import AuditCertifier


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Embroidery Agent · 刺绣针迹生成",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #8B4513;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }
    .stitch-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.15rem;
        background: #f0f0f0;
        border: 1px solid #ddd;
    }
    .format-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "agent" not in st.session_state:
    st.session_state.agent = EmbroideryAgent(enable_audit=True)
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sewing-machine.png", width=64)
    st.title("刺绣针迹生成")
    st.caption("Embroidery Agent v0.2")

    st.divider()

    st.subheader("⚙️ 生成设置")

    max_colors = st.slider("最大颜色数", 2, 12, 6, help="K-means颜色分割的聚类数")
    stitch_density = st.slider("针迹密度", 1, 10, 5, help="值越大针迹越密")
    fill_spacing = st.slider("填充间距 (mm)", 0.5, 5.0, 1.5, step=0.5)

    st.divider()

    st.subheader("📁 导出格式")
    export_pes = st.checkbox("PES (Brother)", value=True)
    export_dst = st.checkbox("DST (Tajima)", value=True)
    export_svg = st.checkbox("SVG (预览)", value=True)

    formats = []
    if export_pes: formats.append("pes")
    if export_dst: formats.append("dst")
    if export_svg: formats.append("svg")
    st.session_state.agent.export_formats = formats or ["pes"]

    st.divider()

    st.subheader("👤 设计师信息")
    designer_id = st.text_input("设计师ID", value="anonymous", placeholder="your name")

    st.divider()

    if st.session_state.history:
        st.subheader(f"📜 生成历史 ({len(st.session_state.history)})")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.caption(f"#{len(st.session_state.history)-i} {h['name']} · {h['stitches']:,}针 · {h['colors']}色")

    st.divider()
    st.caption("Powered by DINOv2 + pyembroidery")
    st.caption("© 2026 Embroidery Agent")


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧵 刺绣针迹自动生成</h1>
    <p>上传设计图 → AI分割 → 针迹规划 → 下载刺绣机文件</p>
</div>
""", unsafe_allow_html=True)

tab_generate, tab_preview, tab_certify, tab_about = st.tabs([
    "🚀 生成针迹", "👁️ 预览", "📜 存证", "ℹ️ 关于"
])


# ============================================================
# Tab 1: Generate
# ============================================================
with tab_generate:
    st.header("上传设计图")

    uploaded = st.file_uploader(
        "支持 PNG / JPG / SVG",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="emb_upload",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.session_state.uploaded_image = img
        st.session_state.uploaded_name = uploaded.name.rsplit(".", 1)[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始设计")
            st.image(img, use_container_width=True)
            st.caption(f"尺寸: {img.width} × {img.height} px")

        with col2:
            st.subheader("颜色分析")
            processor = ImageProcessor(max_colors=max_colors)
            processed = processor.process(img)

            # Color palette display
            palette_html = "<div style='display: flex; gap: 8px; flex-wrap: wrap; margin: 1rem 0;'>"
            for color in processed.color_palette:
                palette_html += f"""
                <div style="text-align: center;">
                    <div style="width: 48px; height: 48px; background: {color.hex};
                                border-radius: 8px; border: 2px solid #ddd;"></div>
                    <small style="color: #666;">{color.name}</small>
                </div>
                """
            palette_html += "</div>"
            st.markdown(palette_html, unsafe_allow_html=True)

            st.metric("检测颜色数", len(processed.color_palette))
            st.metric("分割区域数", len(processed.regions))

        # Region details
        if processed.regions:
            st.subheader("区域分析")
            region_cols = st.columns(min(len(processed.regions), 4))
            for i, region in enumerate(processed.regions):
                with region_cols[i % len(region_cols)]:
                    r, g, b = region.dominant_color
                    st.markdown(f"""
                    <div style="background: rgb({r},{g},{b}); border-radius: 8px;
                                padding: 1rem; color: {'white' if (r+g+b)/3 < 128 else 'black'};">
                        <strong>区域 {i+1}</strong><br>
                        针法: {region.stitch_type.value}<br>
                        面积: {region.area/1000:.1f}K px
                    </div>
                    """, unsafe_allow_html=True)

        # Generate button
        st.divider()
        if st.button("🚀 生成刺绣文件", type="primary", use_container_width=True):
            with st.spinner("🧵 正在生成针迹..."):
                t0 = time.time()
                result = st.session_state.agent.generate(
                    uploaded.name,
                    tempfile.mkdtemp(),
                    st.session_state.uploaded_name,
                )
                dt = (time.time() - t0) * 1000

            st.session_state.last_result = result

            # Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("总针数", f"{result.stitch_plan.total_stitches:,}")
            with m2:
                st.metric("颜色数", result.stitch_plan.total_colors)
            with m3:
                st.metric("区域数", result.regions_count)
            with m4:
                st.metric("尺寸", f"{result.stitch_plan.design_width_mm:.0f}×{result.stitch_plan.design_height_mm:.0f}mm")
            with m5:
                st.metric("耗时", f"{dt:.0f}ms")

            # Download buttons
            st.subheader("📥 下载文件")
            dl_cols = st.columns(len(result.exports))
            for i, exp in enumerate(result.exports):
                with dl_cols[i]:
                    with open(exp.file_path, "rb") as f:
                        data = f.read()
                    st.download_button(
                        label=f"⬇️ {exp.format.upper()} ({exp.file_size_bytes//1024}KB)",
                        data=data,
                        file_name=os.path.basename(exp.file_path),
                        mime={
                            "pes": "application/octet-stream",
                            "dst": "application/octet-stream",
                            "svg": "image/svg+xml",
                        }.get(exp.format, "application/octet-stream"),
                        use_container_width=True,
                    )

            # History
            st.session_state.history.append({
                "name": st.session_state.uploaded_name,
                "stitches": result.stitch_plan.total_stitches,
                "colors": result.stitch_plan.total_colors,
                "time": dt,
            })

            st.success("✅ 生成完成！可以下载文件或前往预览页面")


# ============================================================
# Tab 2: Preview
# ============================================================
with tab_preview:
    st.header("针迹预览")

    if "last_result" not in st.session_state:
        st.warning("请先生成刺绣文件")
    else:
        result = st.session_state.last_result

        # SVG preview
        if result.preview_svg and os.path.exists(result.preview_svg):
            with open(result.preview_svg, "r") as f:
                svg_content = f.read()

            st.subheader("SVG 针迹预览")
            st.components.v1.html(svg_content, height=600, scrolling=True)

            # Download SVG
            st.download_button(
                "📥 下载 SVG 预览",
                data=svg_content.encode(),
                file_name=f"{st.session_state.uploaded_name}_preview.svg",
                mime="image/svg+xml",
            )

        # Stitch block details
        st.subheader("针迹块详情")
        for i, block in enumerate(result.stitch_plan.blocks):
            with st.expander(f"色块 {i+1}: {block.color.name} ({block.stitch_type.value}) — {block.stitch_count} 针"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    - **颜色**: {block.color.hex}
                    - **针法**: {block.stitch_type.value}
                    - **针数**: {block.stitch_count:,}
                    - **尺寸**: {block.bounding_box[2]-block.bounding_box[0]:.1f} × {block.bounding_box[3]-block.bounding_box[1]:.1f} mm
                    """)
                with col2:
                    # Mini preview of stitch path
                    if block.points:
                        import svgwrite
                        dwg = svgwrite.Drawing(f"block_{i}.svg", size=("200px", "200px"))
                        pts = block.points[:200]  # Limit for performance
                        if pts:
                            min_x = min(p.x for p in pts)
                            max_x = max(p.x for p in pts)
                            min_y = min(p.y for p in pts)
                            max_y = max(p.y for p in pts)
                            w = max(max_x - min_x, 1)
                            h = max(max_y - min_y, 1)
                            scale = 180 / max(w, h)
                            ox = 10 - min_x * scale
                            oy = 10 - min_y * scale
                            for j in range(1, len(pts)):
                                if pts[j-1].jump or pts[j].jump:
                                    continue
                                dwg.add(dwg.line(
                                    start=(ox + pts[j-1].x * scale, oy + pts[j-1].y * scale),
                                    end=(ox + pts[j].x * scale, oy + pts[j].y * scale),
                                    stroke=block.color.hex, stroke_width=0.3,
                                ))
                        st.components.v1.html(dwg.tostring(), height=220)


# ============================================================
# Tab 3: Certify
# ============================================================
with tab_certify:
    st.header("设计存证")

    if "last_result" not in st.session_state:
        st.warning("请先生成刺绣文件")
    else:
        result = st.session_state.last_result

        if result.certificate:
            cert = result.certificate

            st.success(f"✅ 设计已存证")
            st.json({
                "设计ID": cert.design_id,
                "设计师": cert.designer_id,
                "针数": cert.stitch_count,
                "颜色数": cert.color_count,
                "格式": cert.file_formats,
                "审计哈希": cert.audit_hash[:32] + "...",
                "链索引": cert.audit_index,
                "创建时间": cert.created_at,
            })

            # Verify
            is_valid = cert.verify()
            if is_valid:
                st.success("✅ 证书完整性验证通过")
            else:
                st.error("❌ 证书验证失败")

            # Download certificate
            st.download_button(
                "📥 下载设计证书 (JSON)",
                data=cert.to_json().encode(),
                file_name=f"{cert.design_id}_certificate.json",
                mime="application/json",
            )
        else:
            st.info("当前生成未启用审计链。重新生成时系统会自动存证。")


# ============================================================
# Tab 4: About
# ============================================================
with tab_about:
    st.header("关于 Embroidery Agent")

    st.markdown("""
    ### 🧵 项目简介

    **Embroidery Agent（刺绣针迹自动生成）** 是一个基于AI的刺绣设计自动化系统，能够将图像自动转换为刺绣机可读的针迹文件。

    ### 🔬 核心技术

    | 技术 | 用途 |
    |------|------|
    | K-means | 颜色分割与区域识别 |
    | DINOv2 | 768维风格指纹提取 |
    | HNSW | 图案库相似检索 |
    | pyembroidery | PES/DST/EXP格式生成 |
    | SHA-256 | 审计链设计存证 |

    ### 🪡 支持针法

    | 针法 | 英文 | 用途 |
    |------|------|------|
    平针 | Running | 轮廓、细节 |
    缎纹针 | Satin | 边框、文字 |
    填充针 | Fill | 实心区域 |
    榻榻米针 | Tatami | 密集填充 |
    链式针 | Chain | 装饰轮廓 |
    锯齿针 | Zigzag | 装饰边框 |
    十字针 | Cross | 计数刺绣 |
    法式结 | French Knot | 点缀 |

    ### 📁 支持格式

    - **PES** — Brother / Baby Lock 刺绣机
    - **DST** — Tajima 通用格式（行业最广泛）
    - **EXP** — Melco 格式
    - **SVG** — 预览与编辑

    ### 📄 开源协议

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
