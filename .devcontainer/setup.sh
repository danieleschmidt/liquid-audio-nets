#\!/bin/bash
set -e

echo "🚀 Setting up liquid-audio-nets development environment..."

# Update package lists
sudo apt-get update

# Install essential development tools
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Install audio development libraries
sudo apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev

# Install ARM embedded toolchain
sudo apt-get install -y \
    gcc-arm-none-eabi \
    gdb-multiarch \
    openocd

# Install Python dependencies
pip install --upgrade pip
pip install -e ".[dev,test,docs]"

# Install Rust components
rustup component add rustfmt clippy
rustup target add thumbv7em-none-eabihf
rustup target add thumbv6m-none-eabi

# Install additional tools
cargo install cargo-audit
cargo install cargo-watch
cargo install cargo-deny

# Set up git hooks (if pre-commit is available)
if command -v pre-commit &> /dev/null; then
    pre-commit install
fi

# Create useful aliases
echo "alias ll='ls -la'" >> ~/.bashrc
echo "alias la='ls -la'" >> ~/.bashrc
echo "alias cargo-check='cargo check --all-targets --all-features'" >> ~/.bashrc
echo "alias cargo-test='cargo test --all-features'" >> ~/.bashrc
echo "alias py-test='pytest tests/ -v'" >> ~/.bashrc

# Set up direnv if available
if command -v direnv &> /dev/null; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
fi

echo "✅ Development environment setup complete\!"
echo ""
echo "🔧 Available commands:"
echo "  make build       - Build all components"
echo "  make test        - Run all tests"
echo "  make lint        - Run linting"
echo "  make docs        - Build documentation"
echo "  make clean       - Clean build artifacts"
echo ""
echo "📚 Next steps:"
echo "  1. Run 'make test' to verify everything works"
echo "  2. Check out docs/DEVELOPMENT.md for development guidelines"
echo "  3. Look at examples/ for getting started"
EOF < /dev/null
