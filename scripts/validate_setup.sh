#!/bin/bash

# llama.cpp Studio - Setup Validation Script
# Verifies all components are properly installed

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  Setup Validation${NC}"
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

print_header

# Track validation results
VALIDATION_PASSED=true

# Check Python
print_info "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION"
else
    print_error "Python 3 not found"
    VALIDATION_PASSED=false
fi

# Check virtual environment
print_info "Checking Python virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    print_success "Virtual environment found"
else
    print_warning "Virtual environment not found (run ./scripts/setup.sh first)"
fi

# Check dependencies
print_info "Checking Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip list 2>/dev/null | grep -q "psutil" && print_success "psutil installed" || print_warning "psutil not installed"
    pip list 2>/dev/null | grep -q "requests" && print_success "requests installed" || print_warning "requests not installed"
else
    print_error "requirements.txt not found"
    VALIDATION_PASSED=false
fi

# Check scripts
print_info "Checking automation scripts..."
SCRIPTS=("setup.sh" "build_llama.cpp.sh" "download_model.sh" "start_server.sh" "stop_server.sh" "test_glm.sh" "benchmark.sh" "env_config.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        [ -x "$script" ] && print_success "$script exists and is executable" || print_warning "$script exists but is not executable"
    else
        print_error "$script not found"
        VALIDATION_PASSED=false
    fi
done

# Check documentation
print_info "Checking documentation files..."
DOCS=("README.md" "PROJECT_OVERVIEW.md" "docs/GLM-4.7_SETUP.md" "docs/TROUBLESHOOTING.md" "docs/BENCHMARKING.md" "docs/QUICKSTART.md")
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        print_success "$doc found"
    else
        print_error "$doc not found"
        VALIDATION_PASSED=false
    fi
done

# Check model
print_info "Checking GLM-4.7 Flash model..."
MODEL_PATH="$HOME/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
if [ -f "$MODEL_PATH" ]; then
    FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    print_success "Model found at $MODEL_PATH ($FILE_SIZE)"
else
    print_warning "Model not found at $MODEL_PATH"
fi

# Check llama.cpp
print_info "Checking llama.cpp..."
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER=$(which llama-server)
    print_success "llama-server found at $LLAMA_SERVER"
else
    print_warning "llama-server not found (run ./scripts/build_llama.cpp.sh first)"
fi

# Check GPU
print_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    GPU_TYPE="nvidia"
elif command -v rocm-smi &> /dev/null; then
    print_success "AMD GPU detected"
    GPU_TYPE="amd"
else
    print_warning "No GPU detected (CPU-only mode possible)"
    GPU_TYPE="cpu"
fi

# Check server availability
print_info "Checking server status..."
if curl -s http://127.0.0.1:11433/v1/models > /dev/null 2>&1; then
    print_success "Server is running"
    SERVER_STATUS="active"
else
    print_warning "Server is not running (start with ./scripts/start_server.sh)"
    SERVER_STATUS="inactive"
fi

# Summary
echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Validation Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}All critical components are present${NC}"
else
    echo -e "${RED}Some components are missing${NC}"
fi

echo ""
echo -e "Environment Status:"
echo -e "  • Python: ${BLUE}${PYTHON_VERSION:-Not found}${NC}"
echo -e "  • GPU Type: ${BLUE}${GPU_TYPE}${NC}"
echo -e "  • Server: ${BLUE}${SERVER_STATUS}${NC}"

echo ""
echo -e "Required Actions:"
echo -e "  1. Run ${BLUE}./scripts/setup.sh${NC} for complete setup"
echo -e "  2. Download model: ${BLUE}./scripts/download_model.sh${NC}"
echo -e "  3. Build llama.cpp: ${BLUE}./scripts/build_llama.cpp.sh${NC}"
echo -e "  4. Start server: ${BLUE}./scripts/start_server.sh${NC}"

echo ""
echo -e "Documentation:"
echo -e "  • Quick Start: ${BLUE}docs/QUICKSTART.md${NC}"
echo -e "• Setup Guide: ${BLUE}docs/GLM-4.7_SETUP.md${NC}"
echo -e "  • Troubleshooting: ${BLUE}docs/TROUBLESHOOTING.md${NC}"
echo -e "  • Benchmarks: ${BLUE}docs/BENCHMARKING.md${NC}"

echo ""
echo -e "${GREEN}================================================================${NC}"
VALIDATION_PASSED && echo -e "${GREEN}Setup validation complete!${NC}" || echo -e "${YELLOW}Validation complete. Some components need attention.${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
