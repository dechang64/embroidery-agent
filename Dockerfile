FROM rust:1.80-slim AS rust-build
WORKDIR /build
COPY Cargo.toml build.rs .
COPY proto/ proto/
COPY src/ src/
RUN cargo build --release 2>/dev/null || true

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl protobuf-compiler && rm -rf /var/lib/apt/lists/*
COPY --from=rust-build /build/target/release/embroidery-agent /usr/local/bin/ 2>/dev/null || true
COPY proto/ proto/
COPY python/ python/
COPY web/ web/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 50051 8080 8501
CMD ["sh", "-c", "embroidery-agent & streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0"]
