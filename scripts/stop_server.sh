#!/bin/bash

# llama.cpp Studio - Server Stop Script
# Stop all llama-server processes gracefully

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}Stopping llama.cpp Server(s)${NC}"
echo ""

# Stop via llama-run
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$PROJECT_ROOT/llama-run" ]; then
    echo "Using llama-run to stop servers..."
    "$PROJECT_ROOT/llama-run" stop-all
    echo ""
fi

# Kill llama-server processes
echo "Stopping llama-server processes..."
PIDS=$(ps aux | grep "[l]lama-server" | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "Found llama-server processes: $PIDS"
    echo "Stopping processes..."
    kill $PIDS 2>/dev/null || true
    sleep 2
    
    # Force kill if still running
    if ps aux | grep "[l]lama-server" > /dev/null; then
        echo "Force stopping remaining processes..."
        kill -9 $PIDS 2>/dev/null || true
        sleep 1
    fi
    
    echo -e "${GREEN}✓${NC} Server processes stopped"
else
    echo "No llama-server processes found"
fi

# Kill processes on port 11433
echo "Checking port 11433..."
PORT_PROCS=$(lsof -ti:11433 2>/dev/null)

if [ -n "$PORT_PROCS" ]; then
    echo "Found processes on port 11433: $PORT_PROCS"
    echo "Stopping processes..."
    kill $PORT_PROCS 2>/dev/null || true
    sleep 1
    
    # Force kill if still running
    if lsof -ti:11433 > /dev/null 2>&1; then
        echo "Force stopping remaining port processes..."
        kill -9 $PORT_PROCS 2>/dev/null || true
        sleep 1
    fi
    
    echo -e "${GREEN}✓${NC} Port 11433 cleared"
else
    echo "Port 11433 is free"
fi

# Check llama-bench processes
echo ""
echo "Checking llama-bench processes..."
LAMBENCH_PROCS=$(ps aux | grep "[l]lama-bench.*server" | awk '{print $2}')

if [ -n "$LAMBENCH_PROCS" ]; then
    echo "Found llama-bench processes: $LAMBENCH_PROCS"
    echo "Stopping processes..."
    kill $LAMBENCH_PROCS 2>/dev/null || true
    sleep 1
    
    echo -e "${GREEN}✓${NC} llama-bench processes stopped"
else
    echo "No llama-bench processes found"
fi

echo ""
echo -e "${GREEN}All llama.cpp Server(s) stopped successfully${NC}"
