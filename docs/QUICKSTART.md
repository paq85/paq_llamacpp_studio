# Quick Start Guide

Get GLM-4.7 Flash running with llama.cpp Studio in minutes!

## Prerequisites

- Python 3.9+
- NVIDIA GPU (8GB+ VRAM) or AMD GPU (ROCm)
- 20GB free disk space
- Stable internet connection (for model download)

## One-Command Setup

```bash
# Run the setup script
./scripts/setup.sh
```

This script will:
1. Check your system requirements
2. Set up Python virtual environment
3. Install all dependencies
4. Build llama.cpp from source
5. Download GLM-4.7 Flash model (~9GB)
6. Create startup and monitoring tools

## After Setup

### Start the Server

```bash
./scripts/start_server.sh
```

Server runs at: http://127.0.0.1:11433

### Run Tests

```bash
# Quick verification
./scripts/test_glm.sh

# Comprehensive benchmarks
./scripts/benchmark.sh
```

## Common Commands

```bash
# Stop the server
./scripts/stop_server.sh

# Check server status
ps aux | grep llama-server

# Monitor GPU
nvidia-smi

# View configuration
cat .env
```

## Testing the Server

```bash
# Test model endpoint
curl http://127.0.0.1:11433/v1/models

# Test completion
curl -X POST http://127.0.0.1:11433/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
    "prompt": "Hello! How are you?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Test chat
curl -X POST http://127.0.0.1:11433/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
    "messages": [
      {"role": "user", "content": "Write a Python function"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Available Tools

### Scripts
- `setup.sh` - Complete environment setup
- `build_llama.cpp.sh` - Build llama.cpp from source
- `download_model.sh` - Download GLM-4.7 Flash model
- `start_server.sh` - Start the server
- `stop_server.sh` - Stop the server
- `test_glm.sh` - Run performance tests
- `benchmark.sh` - Comprehensive benchmarks
- `env_config.sh` - Configure environment

### Python Tools
- `llama-bench` - Full CLI for llama.cpp server
- `llama-run` - TUI server monitoring
- `autotune` - Automatic server optimization

## Documentation

- [README.md](../README.md) - Project overview
- [GLM-4.7_SETUP.md](GLM-4.7_SETUP.md) - Detailed setup guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [BENCHMARKING.md](BENCHMARKING.md) - Performance testing

## Custom Configuration

```bash
# Edit configuration
scripts/env_config.sh

# Or manually edit .env file
nano .env
```

Available settings:
- GPU type (nvidia/amd/cpu)
- Server port
- Context size
- Batch sizes
- Flash attention
- Other performance options

## Getting Help

### Check Server Status

```bash
# Check if server is running
curl http://127.0.0.1:11433/v1/models

# Check GPU metrics
nvidia-smi

# Check system resources
htop
```

### Quick Troubleshooting

```bash
# Stop server if needed
./scripts/stop_server.sh

# Restart server
./scripts/start_server.sh

# Re-run tests
./scripts/test_glm.sh
```

## Performance Tips

### For Maximum Speed
```bash
--batch-size 4096 --ubatch-size 1024 --flash-attn on --kv-unified on
```

### For Low VRAM Usage
```bash
--ctx-size 100000 --batch-size 2048 --ubatch-size 512
```

### For Best Energy Efficiency
```bash
--flash-attn on --kv-unified on --monitor
```

## Example Workflows

### Development Workflow
```bash
# Start server
./scripts/start_server.sh

# Run development tests
python3 test_tokens_per_w.py

# Monitor performance
./llama-run run --port 11433

# Stop when done
./scripts/stop_server.sh
```

### Benchmarking Workflow
```bash
# Full benchmark
./scripts/benchmark.sh

# Custom tests
./scripts/benchmark.sh --prompt-sizes "10k,20k" \
    --output-tokens "256,512"

# Energy test
./scripts/benchmark.sh --task energy
```

### Model Testing Workflow
```bash
# Test individual models
./scripts/test_glm.sh

# Monitor performance
./scripts/start_server.sh

# Compare models
# (Edit model path and re-run tests)
```

## Security Notes

- Server runs locally by default (127.0.0.1)
- No external network exposure
- Model files are local (no cloud storage)
- Virtual environment isolates dependencies

## Performance Expectations

**GLM-4.7 Flash Q4 on modern GPU**:
- Throughput: 150-300+ tokens/sec
- Energy: 3-5 tokens/W
- Latency: 100-500ms
- VRAM: ~9-10GB

## Next Steps

1. ✓ Complete initial setup
2. ✓ Run verification tests
3. ✓ Try basic completions
4. ✓ Explore performance benchmarks
5. ✓ Customize configuration
6. ✓ Read detailed documentation

## Quick Reference

```bash
# Setup
./scripts/setup.sh

# Start
./scripts/start_server.sh

# Test
./scripts/test_glm.sh

# Benchmark
./scripts/benchmark.sh

# Stop
./scripts/stop_server.sh

# Configure
scripts/env_config.sh

# Help
# Check the documentation files
```

## Support

- Read documentation files
- Check troubleshooting guide
- Review test output
- Collect diagnostic information

**Happy benchmarking! 🚀**
