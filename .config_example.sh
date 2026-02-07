#!/bin/bash

# llama.cpp Studio - Configuration Example
# Copy this file to .env and customize for your needs

# Python Virtual Environment
VENV_DIR="$HOME/.pyenv/versions/llama-studio/.venv"

# Model Path
MODEL_PATH="$HOME/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf"

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=11433

# GPU Configuration
GPU_TYPE=nvidia
MIN_VRAM_MB=8192

# llama.cpp Build Configuration
LLAMA_CPP_DIR="$HOME/projects/llama.cpp"

# Build Settings
BUILD_TYPE=release
BUILD_JOBS=$(nproc 2>/dev/null || echo 4)

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
LOG_DIR="$HOME/projects/llama-studio/logs"

# Additional llama.cpp Options
# Use these for advanced configuration
# N_GPU_LAYERS=33
# CONTEXT_SIZE=100000
# BATCH_SIZE=2048

# Environment Variables for llama.cpp
# CUDA_VISIBLE_DEVICES=0
# ROCM_VISIBLE_DEVICES=0
# OMP_NUM_THREADS=8
# MKL_NUM_THREADS=8

