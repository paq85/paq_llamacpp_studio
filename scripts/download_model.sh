#!/bin/bash

# llama.cpp Studio - GLM-4.7 Flash Model Download Script
# Downloads the GLM-4.7-Flash-UD-Q4_K_XL.gguf model

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
    echo -e "${BLUE}  GLM-4.7 Flash Model Downloader${NC}"
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

# Model information
MODEL_NAME="GLM-4.7-Flash-UD-Q4_K_XL.gguf"
MODEL_DIR="$HOME/models"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
MODEL_SIZE="Approximately 9GB"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    print_success "GLM-4.7 Flash model already exists: $MODEL_PATH"
    
    # Get file size
    FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    print_info "Current size: $FILE_SIZE"
    
    echo ""
    print_info "Do you want to re-download the model?"
    read -p "Proceed? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping download"
        exit 0
    fi
fi

# Check if curl or wget is available
print_info "Checking download tools..."
DOWNLOAD_TOOL=""

if command -v curl &> /dev/null; then
    DOWNLOAD_TOOL="curl"
    print_success "curl is available"
elif command -v wget &> /dev/null; then
    DOWNLOAD_TOOL="wget"
    print_success "wget is available"
else
    print_error "No download tool found. Please install curl or wget"
    exit 1
fi

# Ensure model directory exists
print_info "Creating model directory..."
mkdir -p "$MODEL_DIR"
print_success "Model directory created: $MODEL_DIR"

# Model download URLs (using Hugging Face as an example)
# You may want to use your preferred source
MODEL_SOURCES=(
    "https://huggingface.co/THUDM/GLM-4-Flash-9B/resolve/main/GLM-4-Flash-9B-Q4_K_XL.gguf"
    "https://huggingface.co/THUDM/GLM-4-Flash-9B/resolve/main/GLM-4-Flash-9B-Q4_K_M.gguf"
    "https://huggingface.co/THUDM/GLM-4-Flash-9B/resolve/main/GLM-4-Flash-9B-Q5_K_X.gguf"
)

# Select model source
print_info "Available model sources:"
for i in "${!MODEL_SOURCES[@]}"; do
    echo -e "  $((i+1)). ${MODEL_SOURCES[$i]}"
done
echo ""
read -p "Select model source (default: 1): " -r MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

if [ "$MODEL_CHOICE" -ge 1 ] && [ "$MODEL_CHOICE" -le "${#MODEL_SOURCES[@]}" ]; then
    MODEL_URL="${MODEL_SOURCES[$((MODEL_CHOICE-1))]}"
else
    print_error "Invalid selection"
    exit 1
fi

print_info "Selected model source: $MODEL_URL"
echo ""

# Calculate expected download size (approximately)
if command -v curl &> /dev/null; then
    # Get file size info from server
    CONTENT_LENGTH=$(curl -sI "$MODEL_URL" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
    if [ -n "$CONTENT_LENGTH" ] && [ "$CONTENT_LENGTH" != "0" ]; then
        EXPECTED_SIZE=$((CONTENT_LENGTH / 1024 / 1024 / 1024))
        print_info "Expected download size: ~${EXPECTED_SIZE}GB"
    else
        print_warning "Could not determine file size"
    fi
fi

# Confirm download
print_warning "This download will be approximately ${MODEL_SIZE}"
echo ""
print_info "Destination: $MODEL_PATH"
echo ""
read -p "Proceed with download? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Download cancelled"
    exit 0
fi

# Download model
print_info "Starting download..."
PROGRESS_BAR=false

if [ "$DOWNLOAD_TOOL" = "curl" ]; then
    print_info "Using curl for download..."
    
    # Try with resume capability
    if curl -L --progress-bar --retry 3 --retry-delay 5 -o "$MODEL_PATH.part" "$MODEL_URL"; then
        mv "$MODEL_PATH.part" "$MODEL_PATH"
        print_success "Download complete!"
    else
        print_error "Download failed"
        rm -f "$MODEL_PATH.part"
        exit 1
    fi
elif [ "$DOWNLOAD_TOOL" = "wget" ]; then
    print_info "Using wget for download..."
    
    # Try with resume capability
    if wget --continue --tries=3 --wait=5 -O "$MODEL_PATH.part" "$MODEL_URL"; then
        mv "$MODEL_PATH.part" "$MODEL_PATH"
        print_success "Download complete!"
    else
        print_error "Download failed"
        rm -f "$MODEL_PATH.part"
        exit 1
    fi
fi

# Verify download
if [ -f "$MODEL_PATH" ]; then
    FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    print_success "Download verified: $FILE_SIZE"
    
    # Try to validate GGUF file
    print_info "Validating GGUF file..."
    if command -v file &> /dev/null; then
        FILE_TYPE=$(file "$MODEL_PATH" | cut -d: -f2 | xargs)
        if echo "$FILE_TYPE" | grep -i "gguf"; then
            print_success "File type verified as GGUF"
        else
            print_warning "File type might not be GGUF: $FILE_TYPE"
        fi
    fi
else
    print_error "Download verification failed"
    exit 1
fi

# Summary
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Download Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "Model information:"
echo -e "  - Name: ${BLUE}GLM-4.7-Flash-UD-Q4_K_XL.gguf${NC}"
echo -e "  - Size: ${BLUE}$FILE_SIZE${NC}"
echo -e "  - Path: ${BLUE}$MODEL_PATH${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Start the server: ${BLUE}./scripts/start_server.sh${NC}"
echo -e "  2. Run tests: ${BLUE}./scripts/test_glm.sh${NC}"
echo ""
