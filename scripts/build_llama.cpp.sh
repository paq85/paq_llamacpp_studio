#!/bin/bash

# llama.cpp Studio - Build llama.cpp from Source
# This script clones and compiles llama.cpp with GPU support

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  llama.cpp Studio - Build Script${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Print header
print_header

# Check if git is available
print_info "Checking git availability..."
if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git first."
    exit 1
fi

# Determine build configuration based on GPU
LLAMA_BUILD_TYPE="default"

if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected - Building with CUDA support"
    LLAMA_BUILD_TYPE="cuda"
    CUDA_PATH=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    print_success "CUDA detected (version $CUDA_PATH)"
elif command -v rocm-smi &> /dev/null; then
    print_info "AMD GPU detected - Building with ROCm support"
    LLAMA_BUILD_TYPE="amd"
    print_success "ROCm detected"
else
    print_warning "No GPU detected - Building with CPU optimizations"
    LLAMA_BUILD_TYPE="cpu"
fi

# llama.cpp repository
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/build/llama.cpp" && pwd)"

print_info "llama.cpp will be built in: $LLAMA_CPP_DIR"

# Check if llama.cpp already exists
if [ -d "$LLAMA_CPP_DIR/.git" ]; then
    print_info "Updating existing llama.cpp repository..."
    cd "$LLAMA_CPP_DIR"
    git pull
else
    print_info "Cloning llama.cpp repository..."
    mkdir -p "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
    git clone "$LLAMA_CPP_REPO" .
fi

# Configure build
print_info "Configuring build..."

if [ "$LLAMA_BUILD_TYPE" = "cuda" ]; then
    print_info "Building with CUDA support..."
    cmake -B build -DGGML_CUDA=ON -DGGML_VRAM=0 -DGGML_FORCE_CBLAS=ON
elif [ "$LLAMA_BUILD_TYPE" = "amd" ]; then
    print_info "Building with ROCm support..."
    cmake -B build -DGGML_ROCM=ON -DGGML_VRAM=0
else
    print_info "Building with CPU optimizations..."
    cmake -B build -DGGML_VRAM=0 -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_ROCM=OFF
fi

# Build llama.cpp
print_info "Building llama.cpp (this may take several minutes)..."
cmake --build build --config Release -j$(nproc)
print_success "llama.cpp build completed"

# Create symbolic link to llama-server
if [ -f "$LLAMA_CPP_DIR/build/bin/llama-server" ]; then
    ln -sf "$LLAMA_CPP_DIR/build/bin/llama-server" "$PROJECT_ROOT/llama-server"
    print_success "Created symlink: $PROJECT_ROOT/llama-server"
else
    print_error "llama-server binary not found in build directory"
    exit 1
fi

# Verify installation
print_info "Verifying installation..."
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER_PATH=$(which llama-server)
    print_success "llama-server installed at $LLAMA_SERVER_PATH"
    
    # Show version
    if [ -f "$LLAMA_SERVER_PATH" ] && [ "$LLAMA_SERVER_PATH" != "$LLAMA_CPP_DIR/build/bin/llama-server" ]; then
        "$LLAMA_SERVER_PATH" --help | head -3
    fi
else
    print_warning "llama-server not in PATH, check build output"
fi

# Summary
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "llama.cpp has been successfully built with ${LLAMA_BUILD_TYPE^^} support"
echo ""
echo -e "Next steps:"
echo -e "  1. Setup environment: ${BLUE}./scripts/setup.sh${NC}"
echo -e "  2. Download model: ${BLUE}./scripts/download_model.sh${NC}"
echo -e "  3. Start server: ${BLUE}./scripts/start_server.sh${NC}"
echo ""
echo -e "Build configuration:"
echo -e "  - GPU Type: ${BLUE}${LLAMA_BUILD_TYPE^^}${NC}"
echo -e "  - Build Directory: ${BLUE}$LLAMA_CPP_DIR${NC}"
echo -e "  - Binary: ${BLUE}$LLAMA_CPP_DIR/build/bin/llama-server${NC}"
echo ""
