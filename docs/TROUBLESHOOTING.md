# Troubleshooting Guide

Common issues and solutions for llama.cpp Studio with GLM-4.7 Flash.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Server Issues](#server-issues)
3. [Performance Issues](#performance-issues)
4. [Model Issues](#model-issues)
5. [GPU/Driver Issues](#gpu-driver-issues)
6. [Python Dependencies](#python-dependencies)
7. [Network/Download Issues](#network-download-issues)
8. [Advanced Troubleshooting](#advanced-troubleshooting)

## Installation Issues

### Python Not Installed

**Problem**: Python 3 not found

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-venv python3-pip

# macOS
brew install python@3

# Fedora/RHEL
sudo dnf install python3 python3-virtualenv
```

### Virtual Environment Not Creating

**Problem**: `python3 -m venv` fails

**Solutions**:
```bash
# Check Python version
python3 --version  # Must be 3.9+

# Try with explicit Python path
python3 -m venv --clear .venv

# Use system python if needed
/opt/python/*/bin/python3 -m venv .venv
```

### Build Dependencies Missing

**Problem**: CMake not found or other build tools missing

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake git

# macOS
brew install cmake git

# Fedora/RHEL
sudo dnf install cmake gcc-c++ git
```

## Server Issues

### Server Won't Start

**Problem**: Server fails to start with error

**Solutions**:
```bash
# Check for running servers
ps aux | grep llama-server
lsof -i:11433

# Kill existing servers
./scripts/stop_server.sh

# Check model file exists
ls -lh ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf

# Try manual start to see errors
source .venv/bin/activate
python3 -m tools.llama_bench.cli server \
    --model ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf 2>&1
```

### Port Already In Use

**Problem**: Port 11433 is already in use

**Solutions**:
```bash
# Find process using port
lsof -i:11433

# Kill process
kill -9 $(lsof -ti:11433)

# Or change port in config
# Edit .env and change SERVER_PORT
```

### Server Crashes on Startup

**Problem**: Server starts then immediately crashes

**Solutions**:
1. Check VRAM availability: `nvidia-smi`
2. Verify model file integrity: `file ~/models/*.gguf`
3. Check CUDA installation: `nvcc --version`
4. Review error logs in terminal output

### Memory Issues

**Problem**: CUDA out of memory errors

**Solutions**:
```bash
# Reduce context size
--ctx-size 100000

# Reduce batch sizes
--batch-size 2048 --ubatch-size 512

# Reduce GPU layers
--n-gpu-layers 33

# Check available VRAM
nvidia-smi --query-gpu=memory.free,memory.total --format=csv
```

## Performance Issues

### Slow Inference

**Problem**: Processing is slow

**Solutions**:
```bash
# Verify GPU is being used
nvidia-smi -l 1

# Check system resources
htop

# Increase batch sizes
--batch-size 4096 --ubatch-size 1024

# Enable flash attention
--flash-attn on

# Use unified KV cache
--kv-unified on

# Disable unused features
--flash-attn off --kv-unified off
```

### High GPU Temperature

**Problem**: GPU gets too hot

**Solutions**:
```bash
# Monitor temperature
nvidia-smi

# Lower batch sizes (less parallel processing)
--batch-size 2048

# Ensure proper cooling
# Check fan speeds and temperatures

# Reduce GPU utilization
--ctx-size 100000
```

### Low Throughput

**Problem**: Few tokens per second

**Solutions**:
1. Check GPU utilization: `nvidia-smi`
2. Verify CUDA is working: `nvidia-smi dmon`
3. Close other GPU applications
4. Increase CPU cores available
5. Check system RAM usage

### CPU Bottleneck

**Problem**: GPU utilization is low

**Solutions**:
```bash
# Check CPU cores
nproc

# Check system load
uptime

# Optimize batch processing
--batch-size 4096

# Consider CPU-only mode if GPU issues persist
```

## Model Issues

### Model File Not Found

**Problem**: "Model not found" errors

**Solutions**:
```bash
# Check model directory
ls -lh ~/models/

# Set correct path
export MODEL_PATH="$HOME/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf"

# Create symlink if needed
ln -s ~/path/to/model ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf

# Verify file type
file ~/models/*.gguf
```

### Corrupt Model File

**Problem**: Model file appears corrupted

**Solutions**:
```bash
# Check file size
ls -lh ~/models/*.gguf

# Verify GGUF format
xxd ~/models/*.gguf | head -20

# Re-download model
./scripts/download_model.sh

# Alternative: Use different model format
```

### Wrong Model Version

**Problem**: Model is not GLM-4.7 Flash

**Solutions**:
```bash
# Verify model name
ls -lh ~/models/ | grep -i glm

# Check model size (GLM-4.7 Flash ~9GB)
du -sh ~/models/*.gguf

# Download correct model
# GLM-4-Flash-9B: https://huggingface.co/THUDM/GLM-4-Flash-9B
```

## GPU/Driver Issues

### CUDA Not Found

**Problem**: CUDA not detected

**Solutions**:
```bash
# Check CUDA installation
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Install CUDA (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install cuda

# Rebuild llama.cpp
./scripts/build_llama.cpp.sh
```

### ROCm Not Found (AMD)

**Problem**: AMD GPU detected but ROCm not working

**Solutions**:
```bash
# Check ROCm installation
rocm-smi

# Install ROCm (Ubuntu)
wget https://repo.radeon.com/rocm/apt/5.6.1/rocm-release.5.6.1.focal.1.1_all.deb
sudo dpkg -i rocm-release.5.6.1.focal.1.1_all.deb
sudo apt update && sudo apt install rocm-dev

# Rebuild llama.cpp
./scripts/build_llama.cpp.sh
```

### GPU Not Utilized

**Problem**: GPU shows 0% utilization

**Solutions**:
```bash
# Check if CUDA is actually being used
nvidia-smi -l 1

# Look for "python" or "llama-server" processes
nvidia-smi pmon -c 5

# Ensure server is using GPU mode
--flash-attn on

# Try manual build with GPU flags
```

## Python Dependencies

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'psutil'`

**Solutions**:
```bash
# Activate virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Install specific package
pip install psutil requests

# Use fresh virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Version Conflicts

**Problem**: Package version conflicts

**Solutions**:
```bash
# Clean install
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt

# Use virtual environment isolation
# Never install globally when in virtual env

# Check installed packages
pip list | grep -E 'psutil|requests'
```

## Network/Download Issues

### Model Download Fails

**Problem**: Model download fails or is slow

**Solutions**:
```bash
# Check internet connection
ping -c 3 huggingface.co

# Use proxy if needed
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port

# Alternative download methods
# 1. Direct browser download
# 2. Use a different mirror
# 3. Split download into chunks

# Manual verification
curl -I https://huggingface.co/THUDM/GLM-4-Flash-9B/resolve/main/GLM-4-Flash-9B-Q4_K_XL.gguf
```

### Connection Timeout

**Problem**: Connection timeout during download

**Solutions**:
```bash
# Increase timeout
export CURL_TIMEOUT=300

# Try wget instead
wget --continue --tries=10 --wait=5 https://huggingface.co/THUDM/GLM-4-Flash-9B/resolve/main/GLM-4-Flash-9B-Q4_K_XL.gguf

# Use a download manager
```

## Advanced Troubleshooting

### Memory Leak Detection

**Problem**: Server performance degrades over time

**Solutions**:
```bash
# Monitor memory usage
watch -n 1 nvidia-smi

# Check Python memory
watch -n 1 "ps aux | grep llama"

# Monitor logs for memory errors
tail -f logs/server.log

# Restart server regularly
./scripts/stop_server.sh
./scripts/start_server.sh
```

### Process Debugging

**Problem**: Processes not behaving as expected

**Solutions**:
```bash
# Get detailed process info
ps aux | grep llama-server

# Check process status
top -p $(pgrep llama-server)

# Check open files
lsof -p $(pgrep llama-server)

# Core dump analysis
gdb $(which llama-server)
(gdb) run -- --help
(gdb) bt
```

### Logging and Debugging

**Problem**: Need detailed debugging information

**Solutions**:
```bash
# Enable debug logging
export LOG_LEVEL=debug

# Run with verbose output
python3 -m tools.llama_bench.cli server --monitor \
    --verbose

# Save server logs
./scripts/start_server.sh > server_debug.log 2>&1 &

# Analyze logs
grep -i "error\|warn\|fail" server_debug.log
```

### System Resources

**Problem**: System performance affected

**Solutions**:
```bash
# Check all resource usage
htop

# Monitor GPU
nvidia-smi -l 1

# Monitor system load
watch -n 1 uptime

# Check disk usage
du -sh ~/models/ ~/build/

# Clear temp files
find . -name "*.tmp" -delete
```

## Getting Additional Help

### Debug Information Collection

```bash
# Collect diagnostic information
./scripts/setup.sh --diagnostic

# Generate system report
systeminfo > system_report.txt

# Check CUDA versions
nvcc --version
nvidia-smi

# Check Python environment
python3 --version
pip list

# Check GPU
nvidia-smi -L
rocm-smi -l
```

### Common Commands

```bash
# Stop all servers
./scripts/stop_server.sh

# Check server status
ps aux | grep llama-server
lsof -i:11433

# Test connectivity
curl http://127.0.0.1:11433/v1/models

# Monitor resources
htop
nvidia-smi -l 1
```

## Next Steps

If you've tried these solutions and still have issues:

1. Check GitHub Issues for similar problems
2. Review documentation files
3. Collect diagnostic information
4. Ask for help in community forums
5. Consider reinstalling the setup completely

Remember: Always keep your system and dependencies updated.
