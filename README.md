# Embroidery Agent — 刺绣针迹自动生成系统

AI-powered embroidery stitch auto-generation agent with federated learning across workshops.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI (:8501)            │
├─────────────────────────────────────────────────┤
│              Python Client Layer                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Image    │ │  Stitch  │ │  Pattern         │ │
│  │Processor  │ │  Planner │ │  Generator       │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Style   │ │  Audit   │ │  FL Client       │ │
│  │Fingerprint│ │Certifier│ │  (FedAvg)        │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────┤
│              gRPC (:50051) / REST (:8080)        │
├─────────────────────────────────────────────────┤
│              Rust Backend                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  HNSW    │ │  Audit   │ │  Fed Server      │ │
│  │  Index   │ │  Chain   │ │  (Aggregation)   │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────┐ ┌──────────┐                      │
│  │  Vector  │ │  Web     │                      │
│  │  DB      │ │Dashboard │                      │
│  └──────────┘ └──────────┘                      │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Python only (no Rust)
pip install -r requirements.txt
streamlit run web/app.py

# Full stack (Rust + Python)
docker-compose up --build
```

## Features

- **Multi-modal input**: hand-drawn sketches, digital images, vector graphics, text descriptions
- **9 stitch types**: running, satin, fill, chain, zigzag, cross, french_knot, tatami, seed
- **Style fingerprinting**: DINOv2 768-dim feature vectors with HNSW search
- **Pattern export**: PES (Brother), DST (Tajima), EXP (Melco), SVG preview
- **Audit chain**: blockchain-style SHA-256 hash chain for design certification
- **Federated learning**: multi-workshop FedAvg for stitch quality optimization
- **Docker**: full-stack deployment with Rust backend + Streamlit frontend

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Python Client | NumPy, Pillow, OpenCV, pyembroidery |
| gRPC/REST | Tonic (Rust), Axum |
| Vector Index | HNSW (768-dim) |
| Audit | SHA-256 chain + SQLite |
| FL | FedAvg aggregation |
| Proto | protobuf/gRPC |

## Testing

```bash
python -m pytest tests/ -v
```
