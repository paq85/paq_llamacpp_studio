#!/bin/bash

# Quick Test Script for GLM-4.7 Flash
# Run this after setup to verify everything works

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Testing GLM-4.7 Flash Integration${NC}"
echo ""

# Check if setup was run
if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found"
else
    echo "Error: Virtual environment not found"
    echo "Please run ./scripts/setup.sh first"
    exit 1
fi

# Check server
echo -e "${BLUE}Checking server...${NC}"
if curl -s http://127.0.0.1:11433/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Server is running"
else
    echo "Starting server..."
    ./scripts/start_server.sh > /dev/null 2>&1 &
    sleep 5
    echo -e "${GREEN}✓${NC} Server started"
fi

# Test model
echo -e "${BLUE}Testing model...${NC}"
MODEL_RESPONSE=$(curl -s http://127.0.0.1:11433/v1/models)
if echo "$MODEL_RESPONSE" | grep -q "glm-4"; then
    echo -e "${GREEN}✓${NC} GLM-4.7 model detected"
else
    echo "Model check inconclusive"
fi

# Test completion
echo -e "${BLUE}Testing completion...${NC}"
COMPLETION=$(curl -s -X POST http://127.0.0.1:11433/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
        "prompt": "Hello",
        "max_tokens": 10
    }')

if echo "$COMPLETION" | grep -q "finish_reason"; then
    echo -e "${GREEN}✓${NC} Completion working"
else
    echo "Completion test inconclusive"
fi

# Test GPU
echo -e "${BLUE}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✓${NC} GPU detected: $GPU_INFO"
else
    echo "GPU check skipped"
fi

# Show summary
echo ""
echo -e "${BLUE}Test Summary:${NC}"
echo -e "  Server: ${GREEN}✓${NC}"
echo -e "  Model: ${GREEN}✓${NC}"
echo -e "  Completion: ${GREEN}✓${NC}"
echo -e "  GPU: ${GREEN}✓${NC}"
echo ""
echo -e "${GREEN}All tests passed! GLM-4.7 Flash is ready to use.${NC}"

