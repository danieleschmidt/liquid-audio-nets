# Multi-stage build for liquid-audio-nets production deployment
ARG PYTHON_VERSION=3.11

# ==========================================
# Build stage: Compile Rust and C++ components
# ==========================================
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    pkg-config \
    libasound2-dev \
    libpulse-dev \
    portaudio19-dev \
    gcc-arm-none-eabi \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Add embedded targets
RUN rustup target add thumbv7em-none-eabihf thumbv6m-none-eabi

# Set up workspace
WORKDIR /build

# Copy source code
COPY . .

# Build Rust components
RUN cargo build --release --features python-bindings

# Build C++ components
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Build Python package
RUN python3 -m pip install --upgrade pip maturin
RUN maturin build --release --features python-bindings

# ==========================================
# Runtime stage: Minimal production image
# ==========================================
FROM python:${PYTHON_VERSION}-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libasound2 \
    libpulse0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r liquid && useradd -r -g liquid liquid

# Set up application directory
WORKDIR /app
RUN chown liquid:liquid /app

# Copy built artifacts from builder stage
COPY --from=builder /build/target/release/libliquid_audio_nets.so /usr/local/lib/
COPY --from=builder /build/target/wheels/*.whl /tmp/
COPY --from=builder /build/build/libliquid_audio_nets_cpp.so /usr/local/lib/

# Update library cache
RUN ldconfig

# Install Python package
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy configuration and examples
COPY --chown=liquid:liquid examples/ ./examples/
COPY --chown=liquid:liquid models/ ./models/

# Switch to non-root user
USER liquid

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import liquid_audio_nets; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import liquid_audio_nets; print('liquid-audio-nets ready')"]

# ==========================================
# Embedded build stage: Cross-compilation
# ==========================================
FROM builder AS embedded

# Install additional embedded tools
RUN apt-get update && apt-get install -y \
    gdb-multiarch \
    openocd \
    && rm -rf /var/lib/apt/lists/*

# Build for embedded targets
RUN cargo build --target thumbv7em-none-eabihf --release --no-default-features --features embedded
RUN cargo build --target thumbv6m-none-eabi --release --no-default-features --features embedded

# Create embedded artifacts directory
RUN mkdir -p /artifacts/stm32f4 /artifacts/stm32f0 && \
    cp target/thumbv7em-none-eabihf/release/*.a /artifacts/stm32f4/ && \
    cp target/thumbv6m-none-eabi/release/*.a /artifacts/stm32f0/

# Copy headers and documentation
COPY include/ /artifacts/include/
COPY docs/embedded/ /artifacts/docs/

# Create deployment package
RUN tar -czf /artifacts/liquid-audio-nets-embedded.tar.gz -C /artifacts .

# ==========================================
# Development stage: Full development environment
# ==========================================
FROM builder AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    gdb \
    valgrind \
    jupyter \
    pandoc \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Rust development tools
RUN cargo install cargo-watch cargo-audit cargo-deny

# Install Python development dependencies
RUN pip install jupyter jupyterlab tensorboard pre-commit

# Set up pre-commit
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

# Expose development ports
EXPOSE 8888 6006 8080

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
EOF < /dev/null
