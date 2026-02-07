#!/bin/bash

# llama.cpp Studio - Complete Installation Script
# Runs all setup steps in sequence

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Complete Installation Sequence${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Step 1: Setup environment
print_info "Step 1/4: Setting up environment..."
./scripts/setup.sh

# Step 2: Download model
print_info "Step 2/4: Downloading GLM-4.7 Flash model..."
./scripts/download_model.sh

# Step 3: Build llama.cpp
print_info "Step 3/4: Building llama.cpp..."
./scripts/build_llama.cpp.sh

# Step 4: Run validation
print_info "Step 4/4: Running validation..."
./scripts/validate_setup.sh

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Start server: ${BLUE}./scripts/start_server.sh${NC}"
echo -e "  2. Run tests: ${BLUE}./scripts/test_glm.sh${NC}"
echo -e "  3. Explore docs: ${BLUE}docs/QUICKSTART.md${NC}"
echo ""
