# GLM-4.7 Flash - Detailed Setup Guide

This guide provides comprehensive instructions for setting up GLM-4.7 Flash with llama.cpp Studio.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Setup](#quick-setup)
3. [Manual Installation](#manual-installation)
4. [Model Configuration](#model-configuration)
5. [Server Configuration](#server-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Advanced Setup](#advanced-setup)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: 4+ cores (recommended: 8+ cores for parallel processing)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: 
  - NVIDIA: GTX 1060 or better with 8GB+ VRAM (CUDA support)
  - AMD: RX 6000 series or better with 8GB+ VRAM (ROCm support)
- **Storage**: 
  - Model: 10GB free space
  - System: 20GB free space
- **Operating System**: Linux (Ubuntu 20.04+), macOS 11+, or Windows 11 (WSL2 recommended)
- **Python**: 3.9+

### Recommended Configuration

- **CPU**: Modern multi-core processor (12+ cores)
- **RAM**: 32GB minimum (64GB for heavy workloads)
- **GPU**: NVIDIA RTX 3060 or better with 12GB+ VRAM
- **Storage**: NVMe SSD recommended for faster loading
- **Network**: Stable connection (for model downloads)

## Quick Setup

The automated setup script handles all dependencies:

```bash
# Full setup with model download and llama.cpp build
./scripts/setup.sh
```

This will:
1. ✓ Check Python and GPU availability
2. ✓ Create Python virtual environment
3. ✓ Install Python dependencies
4. ✓ Setup model directory
5. ✓ Build llama.cpp from source (if not already built)
6. ✓ Download GLM-4.7 Flash model (optional)
7. ✓ Create configuration files
8. ✓ Create startup scripts

## Manual Installation

### 1. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build llama.cpp

```bash
# Clone and build llama.cpp with GPU support
./scripts/build_llama.cpp.sh
```

### 3. Download GLM-4.7 Flash Model

```bash
# Download the 4-bit quantized model (~9GB)
./scripts/download_model.sh
```

Alternative download locations:
- Hugging Face: https://huggingface.co/THUDM/GLM-4-Flash-9B
- Alternative formats: Q4_K_M, Q5_K_X, etc.

## Model Configuration

### Available GLM-4.7 Flash Models

| Model | Quantization | Size | Quality | Speed |
|-------|--------------|------|---------|-------|
| GLM-4-Flash-9B-Q4_K_XL | 4-bit | ~9GB | High | Fast |
| GLM-4-Flash-9B-Q4_K_M | 4-bit | ~9GB | High | Fast |
| GLM-4-Flash-9B-Q5_K_X | 5-bit | ~11GB | Very High | Medium |

### Model Location

The model should be placed in `~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf`

```bash
# Create model directory
mkdir -p ~/models

# Download model
./scripts/download_model.sh
```

### Model Selection

You can select different model formats by modifying the configuration:

```bash
# Edit environment configuration
scripts/env_config.sh

# Or manually update .env
export MODEL_PATH="$HOME/models/GLM-4.7-Flash-9B-Q5_K_X.gguf"
```

## Server Configuration

### Default Settings

GLM-4.7 Flash is pre-configured with optimal settings:

- **Context Size**: 200K tokens
- **Batch Size**: 4096
- **Micro-batch Size**: 1024
- **Flash Attention**: Enabled
- **Unified KV Cache**: Enabled
- **GPU Layers**: Auto-fit to available VRAM

### Starting the Server

```bash
# Using the setup script
./scripts/start_server.sh

# Or manually
source .venv/bin/activate
python3 -m tools.llama_bench.cli server --monitor \
    --model ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf
```

### Server Ports

- **API**: http://127.0.0.1:11433
- **Model endpoint**: http://127.0.0.1:11433/v1/models
- **Completion endpoint**: http://127.0.0.1:11433/v1/completions
- **Chat endpoint**: http://127.0.0.1:11433/v1/chat/completions

### Custom Configuration

Edit the environment configuration file:

```bash
scripts/env_config.sh
```

Or manually edit `.env`:

```bash
# Server settings
SERVER_HOST=127.0.0.1
SERVER_PORT=11433

# GPU settings
GPU_TYPE=nvidia
CONTEXT_SIZE=200000
BATCH_SIZE=4096
UBATCH_SIZE=1024

# Additional options
FLASH_ATTENTION=on
KV_UNIFIED=on
ENABLE_THINKING=true
```

## Performance Optimization

### GPU Configuration

#### NVIDIA GPUs

```bash
# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# Force CUDA if needed
export FORCE_CUDA=1
```

#### AMD GPUs (ROCm)

```bash
# Set ROCm path
export HIP_VISIBLE_DEVICES=0
export HIP_PATH=/opt/rocm
```

### Memory Optimization

#### VRAM Management

```bash
# Set minimum VRAM required
MIN_VRAM_MB=8192

# Skip VRAM check if needed
--skip-vram-check
```

#### Context Fitting

```bash
# Enable context fitting
--fit on

# Adjust context size for available VRAM
--ctx-size 100000  # For less VRAM
```

### Speed Optimization

```bash
# Batch size adjustments
--batch-size 4096   # Large batch for speed
--ubatch-size 1024  # Smaller micro-batches

# Flash attention for faster inference
--flash-attn on

# Unified KV cache
--kv-unified on
```

## Advanced Setup

### Multiple Models

```bash
# Set up multiple models
export MODEL_1="$HOME/models/GLM-4.7-Flash-9B-Q4_K_XL.gguf"
export MODEL_2="$HOME/models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf"

# Run with different models
python3 -m tools.llama_bench.cli server \
    --model $MODEL_1 --alias glm-4-flash
```

### Custom Chat Templates

```bash
# Use custom Jinja templates
--jinja
--chat-template-kwargs '{"max_length": 200000, "max_new_tokens": 2048}'
```

### Advanced Monitoring

```bash
# Real-time monitoring
./llama-run run --port 11433

# Periodic monitoring
./llama-run run --monitor --interval 1 --duration 300

# JSON output for logging
./llama-run run --json --port 11433
```

### Autotuning

```bash
# Automatic GPU context fitting
python3 -m tools.llama_bench.cli autotune
```

This will:
1. Start the server
2. Run multiple prompt sizes
3. Monitor performance
4. Save optimal settings to `tools/llama_bench/tuned.json`

## Troubleshooting

### Server Won't Start

**Problem**: Server fails to start

**Solutions**:
1. Check if another server is running: `lsof -i:11433`
2. Verify model file exists: `ls -lh ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf`
3. Check VRAM availability: `nvidia-smi` or `rocm-smi`
4. Review logs: Look for error messages in terminal output

### Out of Memory Errors

**Problem**: CUDA out of memory or VRAM errors

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

### Slow Performance

**Problem**: Inference is slow

**Solutions**:
1. Verify GPU is being used: `nvidia-smi`
2. Increase batch sizes
3. Enable flash attention: `--flash-attn on`
4. Use unified KV cache: `--kv-unified on`
5. Close other GPU applications

### Model Not Found

**Problem**: "Model not found" errors

**Solutions**:
```bash
# Verify model path
ls -lh ~/models/

# Set correct model path
export MODEL_PATH="$HOME/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf"

# Check configuration
cat .env | grep MODEL_PATH
```

### Python Dependencies Issues

**Problem**: Import errors or missing packages

**Solutions**:
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# Install fresh dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python3 -c "import psutil, requests; print('Dependencies OK')"
```

### Network Issues (Model Download)

**Problem**: Model download fails

**Solutions**:
1. Check network connection
2. Use alternative download method
3. Manual download from mirror
4. Use proxy if needed: `export http_proxy=http://proxy:port`

## Monitoring and Testing

### Run Tests

```bash
# Quick verification test
./scripts/test_glm.sh

# Comprehensive benchmarking
./scripts/benchmark.sh

# Energy efficiency test
./scripts/benchmark.sh --task energy
```

### Monitor Performance

```bash
# GPU monitoring
nvidia-smi -l 1  # NVIDIA
rocm-smi -d 1    # AMD

# System resources
htop

# llama-run monitoring
./llama-run run --port 11433
```

## Getting Help

For additional support:

- Check existing documentation
- Review GitHub Issues
- Look at troubleshooting guide
- Verify system requirements

## Additional Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GLM-4 Paper](https://arxiv.org/abs/2403.13872)
- [Hugging Face Models](https://huggingface.co/THUDM)

## Next Steps

1. ✓ Complete setup with `./scripts/setup.sh`
2. ✓ Test with `./scripts/test_glm.sh`
3. ✓ Run benchmarks with `./scripts/benchmark.sh`
4. ✓ Customize configuration with `scripts/env_config.sh`
5. ✓ Explore advanced features and optimization techniques
