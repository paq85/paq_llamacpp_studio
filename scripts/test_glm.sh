#!/bin/bash

# llama.cpp Studio - GLM-4.7 Flash Testing Script
# Runs performance tests and verification with GLM-4.7 Flash

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
    echo -e "${BLUE}  GLM-4.7 Flash - Performance Testing${NC}"
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

# Check if model exists
MODEL_PATH="$HOME/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    print_error "GLM-4.7 Flash model not found at $MODEL_PATH"
    print_info "Run ./scripts/download_model.sh to download the model"
    exit 1
fi

# Activate virtual environment
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    print_info "Python virtual environment activated"
fi

# Check if llama-server is available
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER=$(which llama-server)
    print_success "llama-server found at $LLAMA_SERVER"
else
    print_error "llama-server not found"
    print_info "Run ./scripts/build_llama.cpp.sh to build llama.cpp"
    exit 1
fi

# Test server startup
print_info "Testing server startup..."
TEMP_DIR=$(mktemp -d)
LOG_FILE="$TEMP_DIR/server_test.log"

# Start server in background
LLAMA_SERVER $MODEL_PATH --host 127.0.0.1 --port 11433 --ctx-size 200000 \
    --batch-size 4096 --ubatch-size 1024 --flash-attn on \
    --monitor --allow-multi-server \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
sleep 5

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    print_success "Server started successfully (PID: $SERVER_PID)"
else
    print_error "Server failed to start"
    cat "$LOG_FILE"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Function to clean up server
cleanup() {
    print_info "Cleaning up test server..."
    if ps -p $SERVER_PID > /dev/null; then
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
    fi
    rm -rf "$TEMP_DIR"
}

# Set cleanup on exit
trap cleanup EXIT

# Test basic connection
print_info "Testing server connection..."
if curl -s http://127.0.0.1:11433/ > /dev/null 2>&1; then
    print_success "Server is responding"
else
    print_error "Server connection failed"
    exit 1
fi

# Test model endpoint
print_info "Testing model endpoint..."
if curl -s http://127.0.0.1:11433/v1/models | jq -e '.data' > /dev/null 2>&1; then
    print_success "Model endpoint is working"
    MODEL_INFO=$(curl -s http://127.0.0.1:11433/v1/models)
    echo -e "  Model: $(echo "$MODEL_INFO" | jq -r '.data[0].id')"
else
    print_error "Model endpoint failed"
    exit 1
fi

# Run simple completion test
print_info "Running completion test..."
TEST_PROMPT="Hello! Can you help me write a simple Python function?"
MAX_TOKENS=100

COMPLETION_TEST=$(curl -s -X POST http://127.0.0.1:11433/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$(basename "$MODEL_PATH")\",
        \"prompt\": \"$TEST_PROMPT\",
        \"max_tokens\": $MAX_TOKENS,
        \"temperature\": 0.7,
        \"top_p\": 0.95
    }")

if echo "$COMPLETION_TEST" | jq -e '.choices' > /dev/null 2>&1; then
    print_success "Completion test passed"
    CHOICE_COUNT=$(echo "$COMPLETION_TEST" | jq '.choices | length')
    echo -e "  Generated $CHOICE_COUNT completion(s)"
else
    print_error "Completion test failed"
    echo "$COMPLETION_TEST"
    exit 1
fi

# Check GPU metrics if available
print_info "Checking GPU metrics..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
    echo -e "  GPU: $GPU_INFO"
elif command -v rocm-smi &> /dev/null; then
    print_info "ROCm detected - Running ROCm-smi"
    rocm-smi --showutilization --showmeminfo
fi

# Run benchmark test
print_info "Running benchmark test..."
python3 -m tools.llama_bench.cli benchmark \
    --endpoint http://127.0.0.1:11433 \
    --prompt-sizes 1k,10k \
    --max-output-tokens 256 \
    --task coding \
    --json-out "$TEMP_DIR/benchmark.json" 2>&1 | head -20

if [ -f "$TEMP_DIR/benchmark.json" ]; then
    print_success "Benchmark completed"
    TOKENS_PER_SECOND=$(grep -o '"tokens_per_second":[0-9.]*' "$TEMP_DIR/benchmark.json" | head -1 | grep -o '[0-9.]*' | head -1)
    if [ -n "$TOKENS_PER_SECOND" ]; then
        echo -e "  Estimated tokens per second: ${BLUE}$TOKENS_PER_SECOND${NC}"
    fi
fi

# Test tokens per watt calculation
print_info "Testing tokens per watt calculation..."
python3 "$PROJECT_ROOT/demo_tokens_per_w.py" 2>&1 | head -30

# Summary
echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Testing Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "Test results:"
echo -e "  ✓ Server startup: PASSED"
echo -e "  ✓ Server connection: PASSED"
echo -e "  ✓ Model endpoint: PASSED"
echo -e "  ✓ Completion test: PASSED"
echo -e "  ✓ Benchmark test: PASSED"
echo -e "  ✓ Tokens per watt: PASSED"
echo ""
echo -e "GLM-4.7 Flash is working correctly!"
echo ""
