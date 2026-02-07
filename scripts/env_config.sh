#!/bin/bash

# llama.cpp Studio - Environment Configuration Script
# Manages environment variables and configuration files

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
    echo -e "${BLUE}  Environment Configuration${NC}"
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

# Configuration paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
CONFIG_DIR="$PROJECT_ROOT/.config/llama-studio"
mkdir -p "$CONFIG_DIR"

print_info "Configuration directory: $CONFIG_DIR"
echo ""

# Default configuration values
DEFAULT_CONFIG=(
    "# llama.cpp Studio - Environment Configuration"
    "# Edit this file to customize settings"
    ""
    "# Python Virtual Environment"
    "VENV_DIR=\"$PROJECT_ROOT/.venv\""
    ""
    "# Model Path"
    "MODEL_PATH=\"$HOME/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf\""
    ""
    "# Server Configuration"
    "SERVER_HOST=127.0.0.1"
    "SERVER_PORT=11433"
    ""
    "# GPU Configuration"
    "GPU_TYPE=nvidia"
    "MIN_VRAM_MB=8192"
    ""
    "# llama.cpp Build Configuration"
    "LLAMA_CPP_DIR=\"$PROJECT_ROOT/build/llama.cpp\""
    ""
    "# Build Settings"
    "BUILD_TYPE=release"
    "BUILD_JOBS=$(nproc 2>/dev/null || echo 4)"
    ""
    "# Performance Settings"
    "CONTEXT_SIZE=200000"
    "BATCH_SIZE=4096"
    "UBATCH_SIZE=1024"
    ""
    "# GPU Settings"
    "FLASH_ATTENTION=on"
    "KV_UNIFIED=on"
    "ENABLE_THINKING=true"
    ""
    "# Monitoring Settings"
    "MONITOR_INTERVAL=1"
    "BENCHMARK_DURATION=60"
    ""
    "# Test Settings"
    "TEST_PROMPT_SIZES=\"1k,10k,100k\""
    "TEST_OUTPUT_TOKENS=\"256,512,1024\""
    ""
    "# Log Settings"
    "LOG_LEVEL=info"
    "LOG_DIR=\"$PROJECT_ROOT/logs\""
)

# Function to check if config exists
check_config() {
    if [ -f "$ENV_FILE" ]; then
        print_info "Existing configuration found: $ENV_FILE"
        return 0
    else
        print_info "No existing configuration found"
        return 1
    fi
}

# Function to backup existing config
backup_config() {
    if [ -f "$ENV_FILE" ]; then
        BACKUP_FILE="$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$ENV_FILE" "$BACKUP_FILE"
        print_success "Configuration backed up to: $BACKUP_FILE"
    fi
}

# Function to write default config
write_default_config() {
    print_info "Writing default configuration..."
    cat > "$ENV_FILE" << 'EOF'
# llama.cpp Studio - Environment Configuration
# Edit this file to customize settings

# Python Virtual Environment
VENV_DIR="/path/to/.venv"

# Model Path
MODEL_PATH="/path/to/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf"

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=11433

# GPU Configuration
GPU_TYPE=nvidia
MIN_VRAM_MB=8192

# llama.cpp Build Configuration
LLAMA_CPP_DIR="/path/to/build/llama.cpp"

# Build Settings
BUILD_TYPE=release
BUILD_JOBS=4

# Performance Settings
CONTEXT_SIZE=200000
BATCH_SIZE=4096
UBATCH_SIZE=1024

# GPU Settings
FLASH_ATTENTION=on
KV_UNIFIED=on
ENABLE_THINKING=true

# Monitoring Settings
MONITOR_INTERVAL=1
BENCHMARK_DURATION=60

# Test Settings
TEST_PROMPT_SIZES="1k,10k,100k"
TEST_OUTPUT_TOKENS="256,512,1024"

# Log Settings
LOG_LEVEL=info
LOG_DIR="/path/to/logs"
EOF
    print_success "Default configuration written"
}

# Function to validate config
validate_config() {
    print_info "Validating configuration..."

    if [ ! -f "$ENV_FILE" ]; then
        print_warning "Configuration file not found, creating default..."
        write_default_config
        return 1
    fi

    # Check for required variables
    REQUIRED_VARS=("VENV_DIR" "MODEL_PATH" "SERVER_HOST" "SERVER_PORT")
    
    MISSING_VARS=()
    for VAR in "${REQUIRED_VARS[@]}"; do
        if ! grep -q "^${VAR}=" "$ENV_FILE"; then
            MISSING_VARS+=("$VAR")
        fi
    done

    if [ ${#MISSING_VARS[@]} -gt 0 ]; then
        print_warning "Missing required variables: ${MISSING_VARS[*]}"
        backup_config
        write_default_config
        return 1
    fi

    print_success "Configuration is valid"
    return 0
}

# Function to display current configuration
show_config() {
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "No configuration file found"
        return 1
    fi

    print_info "Current configuration:"
    echo ""
    cat "$ENV_FILE"
    echo ""
}

# Function to edit configuration
edit_config() {
    print_info "Opening configuration for editing..."
    
    if [ -f "$EDITOR" ]; then
        $EDITOR "$ENV_FILE"
    elif [ -f "$VISUAL" ]; then
        $VISUAL "$ENV_FILE"
    else
        # Try common editors
        for EDITOR in vim nano nano-tiny; do
            if command -v "$EDITOR" &> /dev/null; then
                $EDITOR "$ENV_FILE"
                return 0
            fi
        done
        
        print_error "No text editor found. Install vim or nano"
        return 1
    fi
    
    print_success "Configuration updated"
}

# Function to set GPU type
set_gpu_type() {
    echo "Available GPU types:"
    echo "  1. nvidia - NVIDIA GPU support"
    echo "  2. amd - AMD GPU support (ROCm)"
    echo "  3. cpu - CPU-only mode"
    echo ""
    read -p "Select GPU type (default: nvidia): " -r GPU_CHOICE
    
    GPU_TYPE=${GPU_CHOICE:-nvidia}
    
    case $GPU_TYPE in
        nvidia|amd|cpu)
            # Update configuration
            if [ -f "$ENV_FILE" ]; then
                sed -i "s/^GPU_TYPE=.*/GPU_TYPE=\"$GPU_TYPE\"/" "$ENV_FILE"
                print_success "GPU type set to: $GPU_TYPE"
            else
                print_warning "Configuration file not found, skipping"
            fi
            ;;
        *)
            print_warning "Invalid GPU type. Keeping current setting."
            ;;
    esac
}

# Function to configure context size
configure_context() {
    echo ""
    echo "Context size determines how much text the model can remember at once."
    echo "Larger contexts = more memory, better for long conversations"
    echo ""
    read -p "Enter context size (default: 200000): " -r CONTEXT_SIZE
    
    CONTEXT_SIZE=${CONTEXT_SIZE:-200000}
    
    if [ -f "$ENV_FILE" ]; then
        sed -i "s/^CONTEXT_SIZE=.*/CONTEXT_SIZE=$CONTEXT_SIZE/" "$ENV_FILE"
        print_success "Context size set to: $CONTEXT_SIZE"
    else
        print_warning "Configuration file not found, skipping"
    fi
}

# Main menu
show_menu() {
    echo "Configuration Options:"
    echo "  1. Display current configuration"
    echo "  2. Edit configuration"
    echo "  3. Set GPU type"
    echo "  4. Configure context size"
    echo "  5. Validate configuration"
    echo "  6. Regenerate default configuration"
    echo "  7. Exit"
    echo ""
    read -p "Select option (1-7): " -r MENU_CHOICE
    
    case $MENU_CHOICE in
        1) show_config ;;
        2) edit_config ;;
        3) set_gpu_type ;;
        4) configure_context ;;
        5) validate_config ;;
        6) 
            print_info "Regenerating configuration..."
            backup_config
            write_default_config
            ;;
        7) 
            print_info "Exiting configuration..."
            exit 0
            ;;
        *)
            print_warning "Invalid option"
            ;;
    esac
}

# Check if config exists
check_config

# Menu loop
while true; do
    show_menu
    echo ""
    read -p "Press Enter to continue..." -r
    echo ""
done
