# LlamaCPP Studio

A comprehensive toolkit for working with llama.cpp and GLM-4.7 Flash models. This project provides:

- **llama-bench**: CLI tool for running llama.cpp servers, monitoring resources, and benchmarking prompt performance
- **llama-run**: Lightweight TUI for running llama.cpp servers with real-time monitoring
- **Performance monitoring**: CPU, RAM, GPU (NVIDIA/AMD), and power/energy metrics
- **Autotuning**: Automatic server configuration optimization
- **GLM-4.7 Flash integration**: Pre-configured settings for optimal GLM-4.7 Flash performance

## Quick Start

```bash
# Setup the environment and download GLM-4.7 Flash model
./scripts/setup.sh

# Start the llama.cpp server
./scripts/start_server.sh

# Run benchmarking tests
./scripts/test_glm.sh

# Or use the Python tools
python3 -m tools.llama_bench.cli server --monitor --model ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf
```

## Features

### Built-in Tools

- **llama-bench**: Complete CLI for llama.cpp server management and benchmarking
- **llama-run**: Htop-like TUI for monitoring server performance
- **Autotune**: Automatic GPU context fitting and optimization
- **Benchmarking**: Prompt size analysis and performance metrics

### Monitoring Capabilities

- CPU usage and frequency
- Memory consumption
- GPU utilization, power, temperature, and VRAM
- CPU package power and system energy (RAPL)
- Token processing rates and throughput

### GLM-4.7 Flash Optimizations

- Pre-configured context fitting settings
- Flash attention enabled by default
- Unified KV cache configuration
- Optimized batch sizes
- Energy efficiency monitoring

## Project Structure

```
.
├── scripts/                # Automation scripts for setup and management
│   ├── setup.sh           # Environment and model setup
│   ├── build_llama.cpp.sh # llama.cpp build automation
│   ├── download_model.sh  # GLM-4.7 Flash model download
│   ├── start_server.sh    # Server startup script
│   ├── stop_server.sh     # Server shutdown script
│   └── test_glm.sh        # GLM-4.7 testing
├── tools/                 # Python tooling
│   ├── llama_bench/       # Main llama.cpp CLI and utilities
│   └── llama_run/         # TUI server monitoring
├── tests/                 # Test suite
└── docs/                  # Documentation
    ├── GLM-4.7_SETUP.md    # GLM-4.7 specific setup guide
    ├── TROUBLESHOOTING.md  # Common issues and solutions
    └── BENCHMARKING.md     # Performance testing guide
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (NVIDIA preferred) or ROCm-capable GPU (AMD)
- 8GB+ VRAM (4-bit models can work with 4GB+)
- 16GB+ RAM recommended
- bash shell for automation scripts

## Installation

The automated setup script handles all dependencies:

```bash
# Full setup with model download and llama.cpp build
./scripts/setup.sh
```

Or manually if you prefer:

```bash
# Python dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# llama.cpp (if not already built)
./scripts/build_llama.cpp.sh

# GLM-4.7 Flash model
./scripts/download_model.sh
```

## Usage Examples

### Start Server with Monitoring

```bash
# Using the wrapper script
./scripts/start_server.sh

# Using llama-bench directly
python3 -m tools.llama_bench.cli server --monitor --model ~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf
```

### Benchmarking

```bash
# Quick benchmark
./scripts/benchmark.sh

# Using llama-bench CLI
python3 -m tools.llama_bench.cli benchmark --prompt-sizes 1k,10k,100k --max-output-tokens 1024
```

### Monitor Running Server

```bash
# Using llama-run TUI
./llama-run run --port 11433

# Simple monitoring via HTTP
python3 -m tools.llama_bench.cli monitor --port 11433 --interval 1
```

## Model Information

**GLM-4.7 Flash** - Optimized for fast inference and energy efficiency

- **Model**: GLM-4.7-Flash-UD-Q4_K_XL.gguf
- **Size**: ~4-bit quantized (~9GB)
- **Performance**: Excellent speed/quality balance
- **Use case**: General purpose AI assistant, code completion, and reasoning

## Configuration

Common settings are pre-configured in the automation scripts:

- Context size: 200K
- Batch size: 4096
- Micro-batch size: 1024
- Flash attention: Enabled
- Unified KV cache: Enabled
- GPU layers: Auto-fit

Edit `scripts/start_server.sh` or create `~/.config/llama-studio/config.sh` to customize.

## Troubleshooting

For common issues and solutions, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Contributing

This project is actively maintained for GLM-4.7 Flash integration and llama.cpp performance optimization.

## License

See individual tool files and dependencies for license information.

## Support

- GitHub Issues: [Project Issues Page]
- Documentation: [Project Wiki Pages]
