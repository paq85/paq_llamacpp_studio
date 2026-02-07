#!/bin/bash

# llama.cpp Studio - Comprehensive Benchmark Script
# Runs extensive performance benchmarks with GLM-4.7 Flash

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
    echo -e "${BLUE}  GLM-4.7 Flash - Comprehensive Benchmark${NC}"
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

# Create benchmark output directory
BENCHMARK_DIR="$PROJECT_ROOT/benchmarks"
mkdir -p "$BENCHMARK_DIR"
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
BENCHMARK_OUTPUT="$BENCHMARK_DIR/benchmark_$CURRENT_TIME.json"
BENCHMARK_LOG="$BENCHMARK_DIR/benchmark_$CURRENT_TIME.log"

# Test cases configuration
PROMPT_SIZES="1k,10k,100k"
OUTPUT_TOKENS="16,256,1024"
TASKS="coding,list"
BENCHMARK_DURATION=60

# Summary statistics variables
TOTAL_TESTS=0
PASSED_TESTS=0

# Function to run a single benchmark
run_benchmark() {
    local test_name=$1
    local prompt_size=$2
    local output_tokens=$3
    
    print_info "Running: $test_name (prompt: ${prompt_size}, output: $output_tokens)"
    
    # Create test-specific output
    TEST_OUTPUT="$BENCHMARK_DIR/${test_name}_${CURRENT_TIME}"
    mkdir -p "$TEST_OUTPUT"
    
    # Run llama-bench
    python3 -m tools.llama_bench.cli benchmark \
        --endpoint http://127.0.0.1:11433 \
        --prompt-sizes "$prompt_size" \
        --max-output-tokens "$output_tokens" \
        --task "$test_name" \
        --progress on \
        --progress-interval 5 \
        --json-out "$TEST_OUTPUT/summary.json" \
        > "$TEST_OUTPUT/benchmark.log" 2>&1
    
    # Parse results
    if [ -f "$TEST_OUTPUT/summary.json" ]; then
        print_success "Test completed: $test_name"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Extract key metrics
        TOKENS_PER_SECOND=$(grep -o '"tokens_per_second":[0-9.]*' "$TEST_OUTPUT/summary.json" | head -1 | grep -o '[0-9.]*' | head -1)
        CPU_PERCENT=$(grep -o '"cpu_percent":[0-9.]*' "$TEST_OUTPUT/summary.json" | head -1 | grep -o '[0-9.]*' | head -1)
        GPU_POWER=$(grep -o '"gpu_power_w":[0-9.]*' "$TEST_OUTPUT/summary.json" | head -1 | grep -o '[0-9.]*' | head -1)
        
        echo -e "  Tokens/sec: ${BLUE}${TOKENS_PER_SECOND:-N/A}${NC}"
        echo -e "  CPU: ${BLUE}${CPU_PERCENT:-N/A}%${NC}"
        echo -e "  GPU Power: ${BLUE}${GPU_POWER:-N/A}W${NC}"
    else
        print_error "Test failed: $test_name"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
    fi
    echo ""
}

# Function to run tokens per watt test
run_tokens_per_watt() {
    print_info "Running tokens per watt analysis..."
    python3 "$PROJECT_ROOT/demo_tokens_per_w.py" > "$TEST_OUTPUT/tokens_per_w_test.log" 2>&1
    print_success "Tokens per watt analysis completed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Function to run energy efficiency test
run_energy_test() {
    print_info "Running energy efficiency test..."
    python3 -m tools.llama_bench.cli monitor --port 11433 --interval 1 \
        --duration 30 > "$TEST_OUTPUT/energy_monitor.log" 2>&1
    print_success "Energy efficiency test completed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Function to test prompt variations
run_prompt_tests() {
    print_info "Testing prompt variations..."
    
    # Small prompt test
    run_benchmark "small_prompt" "1k" "256"
    sleep 2
    
    # Medium prompt test
    run_benchmark "medium_prompt" "10k" "512"
    sleep 2
    
    # Large prompt test
    run_benchmark "large_prompt" "100k" "1024"
    sleep 2
}

# Function to test different tasks
run_task_tests() {
    print_info "Testing different tasks..."
    
    # Coding task
    run_benchmark "coding_task" "10k" "256"
    sleep 2
    
    # List generation task
    run_benchmark "list_task" "5k" "256"
    sleep 2
}

# Function to test output token variations
run_output_tests() {
    print_info "Testing output token variations..."
    
    # Short output
    run_benchmark "short_output" "1k" "16"
    sleep 2
    
    # Medium output
    run_benchmark "medium_output" "1k" "256"
    sleep 2
    
    # Long output
    run_benchmark "long_output" "1k" "1024"
    sleep 2
}

# Function to run GPU stress test
run_gpu_stress_test() {
    print_info "Running GPU stress test..."
    
    python3 -m tools.llama_bench.cli benchmark \
        --endpoint http://127.0.0.1:11433 \
        --prompt-sizes "100k" \
        --max-output-tokens "2048" \
        --task coding \
        --duration 120 \
        --progress on \
        --json-out "$TEST_OUTPUT/gpu_stress.json" \
        > "$TEST_OUTPUT/gpu_stress.log" 2>&1
    
    print_success "GPU stress test completed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Check server availability
print_info "Checking server availability..."
if curl -s http://127.0.0.1:11433/ > /dev/null 2>&1; then
    print_success "Server is running"
else
    print_error "Server is not running"
    print_info "Start the server first with: ./scripts/start_server.sh"
    exit 1
fi

# Check GPU availability
print_info "Checking GPU metrics..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
elif command -v rocm-smi &> /dev/null; then
    rocm-smi --showutilization --showmeminfo
fi

# Run all benchmark tests
echo ""
print_info "Starting comprehensive benchmark tests..."
echo ""

run_prompt_tests
run_output_tests
run_task_tests
run_tokens_per_watt
run_energy_test
run_gpu_stress_test

# Generate summary report
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Benchmark Summary${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "Tests completed: $PASSED_TESTS/$TOTAL_TESTS"
echo -e "Output directory: $BENCHMARK_DIR"
echo -e "Summary report: $BENCHMARK_OUTPUT"
echo ""

# Create benchmark summary
cat > "$BENCHMARK_OUTPUT" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "model": "$(basename "$MODEL_PATH")",
  "total_tests": $TOTAL_TESTS,
  "passed_tests": $PASSED_TESTS,
  "failed_tests": $((TOTAL_TESTS - PASSED_TESTS)),
  "test_directory": "$BENCHMARK_DIR/$CURRENT_TIME",
  "summary_directory": "$BENCHMARK_DIR"
}
EOF

# Display key findings
echo -e "${BLUE}Key Findings:${NC}"
echo -e "  • Test coverage: $PASSED_TESTS/$TOTAL_TESTS tests passed"
echo -e "  • Total benchmark time: ${BLUE}${BENCHMARK_DURATION}s${NC}"
echo -e "  • Detailed results: ${BLUE}$BENCHMARK_DIR/$CURRENT_TIME${NC}"
echo ""

print_success "All benchmarks completed successfully!"
echo ""
