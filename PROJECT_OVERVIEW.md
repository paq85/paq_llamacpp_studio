# Project Overview

Complete automation and documentation for llama.cpp Studio with GLM-4.7 Flash.

## Project Structure

```
paq_llamacpp_studio/
├── README.md                          # Main project documentation
├── requirements.txt                    # Python dependencies
├── .env                                # Environment configuration (auto-generated)
├── scripts/                            # Automation scripts
│   ├── setup.sh                       # Complete environment setup
│   ├── build_llama.cpp.sh             # Build llama.cpp from source
│   ├── download_model.sh              # Download GLM-4.7 Flash model
│   ├── start_server.sh                # Start llama-server
│   ├── stop_server.sh                 # Stop llama-server
│   ├── test_glm.sh                    # Performance verification tests
│   ├── benchmark.sh                   # Comprehensive benchmarks
│   └── env_config.sh                  # Environment configuration menu
├── tools/                             # Python tooling
│   ├── llama_bench/                   # Main llama.cpp CLI
│   │   ├── cli.py                     # Main command-line interface
│   │   ├── benchmark.py               # Benchmark functionality
│   │   ├── autotune.py                # Auto-configuration tool
│   │   ├── logperf.py                 # Logging utilities
│   │   ├── proxy.py                   # HTTP proxy handling
│   │   ├── tui.py                     # Terminal UI
│   │   ├── utils.py                   # Helper functions
│   │   ├── __init__.py                # Package initialization
│   │   └── requirements.txt           # Tool dependencies
│   └── llama_run/                     # Server monitoring tool
│       ├── cli.py                     # Command-line interface
│       ├── logperf.py                 # Performance logging
│       ├── proxy.py                   # HTTP proxy
│       ├── tui.py                     # Terminal UI
│       ├── README.md                  # Tool documentation
│       └── __init__.py                # Package initialization
├── tests/                             # Test suite
│   ├── acceptance/                    # Acceptance tests
│   ├── bin/                           # Test binaries
│   ├── fake_llama_server.py           # Test server stub
│   └── fake_llama_server.py           # Acceptance tests
├── docs/                              # Documentation
│   ├── GLM-4.7_SETUP.md               # Detailed GLM-4.7 setup guide
│   ├── TROUBLESHOOTING.md             # Troubleshooting guide
│   ├── BENCHMARKING.md                # Performance benchmarking guide
│   └── QUICKSTART.md                  # Quick start guide
├── demo_tokens_per_w.py               # Tokens per watt demonstration
├── test_tokens_per_w.py               # Performance test suite
├── llama-bench                        # llama-bench binary
├── llama-run                          # llama-run binary
├── .venv/                             # Python virtual environment
└── build/                             # Build directories
    └── llama.cpp/                     # llama.cpp source/build
```

## Script Overview

### Setup Scripts

**scripts/setup.sh** (7.3K)
- Full environment configuration
- Python virtual environment setup
- llama.cpp build preparation
- Model directory creation
- Configuration file generation
- Dependencies installation

**scripts/build_llama.cpp.sh** (4.4K)
- Clones and builds llama.cpp from source
- GPU detection (CUDA/ROCm)
- Optimized build configurations
- Creates system symbolic links

**scripts/download_model.sh** (5.6K)
- Downloads GLM-4.7 Flash model
- Automatic model source selection
- Resume capability for downloads
- File integrity verification

### Server Management

**scripts/start_server.sh** (Auto-generated)
- Starts llama-server with optimal settings
- GLM-4.7 Flash pre-configurations
- Real-time monitoring enabled
- GPU optimizations applied

**scripts/stop_server.sh** (2.2K)
- Gracefully stops all server processes
- Port cleanup
- Process management
- Error handling

### Testing & Benchmarking

**scripts/test_glm.sh** (5.6K)
- Server connectivity tests
- Model endpoint verification
- Completion generation tests
- GPU metrics checking
- Performance baseline creation

**scripts/benchmark.sh** (7.6K)
- Comprehensive performance testing
- Prompt size variations
- Output token tests
- Task-based benchmarks
- Energy efficiency monitoring
- GPU stress testing

### Configuration

**scripts/env_config.sh** (7.7K)
- Environment variable management
- Configuration file editing
- GPU type selection
- Context size configuration
- Performance tuning options

## Documentation Files

**README.md** (Project Overview)
- Project features and capabilities
- Quick start instructions
- Tool descriptions
- Usage examples
- Configuration options

**docs/GLM-4.7_SETUP.md** (GLM-4.7 Setup)
- System requirements
- Installation procedures
- Model configuration
- Server setup
- Performance optimization
- Advanced configurations

**docs/TROUBLESHOOTING.md** (Common Issues)
- Installation problems
- Server issues
- Performance problems
- Model issues
- GPU/driver problems
- Advanced debugging

**docs/BENCHMARKING.md** (Performance Testing)
- Benchmark types
- Performance metrics
- Interpretation guide
- Optimization techniques
- Reporting methods
- Reference values

**docs/QUICKSTART.md** (Getting Started)
- One-command setup
- Common commands
- Testing examples
- Workflow examples
- Quick reference

## Python Tools

### llama-bench
Full-featured CLI for llama.cpp server management:
- Server control (start/stop/monitor)
- Resource monitoring (CPU, RAM, GPU)
- Performance benchmarking
- Auto-tuning capabilities
- Real-time metrics display

### llama-run
Lightweight server monitoring with TUI:
- Htop-like interface
- Token processing tracking
- Resource monitoring
- Simple scrolling output mode

## Key Features

### Automated Setup
- One-command environment setup
- Automatic dependency installation
- GPU detection and configuration
- Model preparation

### Performance Monitoring
- GPU utilization tracking
- Power consumption monitoring
- Energy efficiency metrics
- Temperature monitoring

### Comprehensive Testing
- Quick verification tests
- Extensive benchmark suites
- Energy efficiency analysis
- Performance comparisons

### Configuration Management
- Environment variable setup
- Multiple configuration profiles
- Auto-tuning capabilities
- Easy customization

### Documentation
- Complete setup guides
- Troubleshooting references
- Performance analysis guides
- Quick start instructions

## GLM-4.7 Flash Integration

### Optimized Settings
- Context size: 200K tokens
- Batch sizes: 4096/1024
- Flash attention: Enabled
- Unified KV cache: Enabled
- GPU layer auto-fitting

### Model Support
- GLM-4.7-Flash-UD-Q4_K_XL.gguf
- Alternative formats (Q4_K_M, Q5_K_X)
- Flexible model path configuration
- Multi-model support

## Workflow

### Initial Setup
```bash
./scripts/setup.sh
```

### Daily Development
```bash
./scripts/start_server.sh
# Use tools...
./scripts/stop_server.sh
```

### Performance Testing
```bash
./scripts/test_glm.sh
./scripts/benchmark.sh
```

### Configuration
```bash
scripts/env_config.sh
```

## Technical Specifications

### System Requirements
- **Python**: 3.9+
- **GPU**: NVIDIA CUDA or AMD ROCm
- **RAM**: 16GB minimum (32GB recommended)
- **VRAM**: 8GB minimum (NVIDIA/AMD)
- **Storage**: 10GB model space + 20GB system

### Build Configuration
- **llama.cpp**: Latest stable version
- **GPU Support**: CUDA/ROCm/CPU modes
- **Flash Attention**: Enabled
- **KV Cache**: Unified
- **Build Type**: Release with optimizations

## Version Information

**Project Version**: 1.0.0
**GLM-4.7 Flash**: Latest release
**llama.cpp**: Latest stable build
**Python**: 3.9+

## Contributing

This project is maintained for GLM-4.7 Flash integration and llama.cpp performance optimization.

## License

See individual tool files and dependencies for license information.

## Support

- Read documentation files
- Check troubleshooting guide
- Review performance benchmarks
- Verify system requirements

## Next Steps

1. ✓ Review README.md
2. ✓ Read docs/QUICKSTART.md
3. ✓ Run ./scripts/setup.sh
4. ✓ Read docs/GLM-4.7_SETUP.md
5. ✓ Try ./scripts/test_glm.sh
6. ✓ Run ./scripts/benchmark.sh

---

**Complete automation for llama.cpp Studio with GLM-4.7 Flash! 🚀**
