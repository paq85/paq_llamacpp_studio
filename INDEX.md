# llama.cpp Studio - Master Index

## Quick Navigation

### Getting Started
- Quick Start: docs/QUICKSTART.md
- Project Overview: README.md
- Setup Guide: docs/GLM-4.7_SETUP.md

### Documentation
- Project Structure: PROJECT_OVERVIEW.md
- Troubleshooting: docs/TROUBLESHOOTING.md
- Benchmarks: docs/BENCHMARKING.md

## Automation Scripts

### Setup & Installation
./scripts/setup.sh           - Complete environment setup
./scripts/install_all.sh     - Run all setup steps
./scripts/validate_setup.sh  - Check installation

### Server Management
./scripts/start_server.sh    - Start the server
./scripts/stop_server.sh     - Stop the server

### Testing & Benchmarking
./scripts/test_glm.sh        - Quick verification tests
./scripts/benchmark.sh       - Comprehensive benchmarks
quick_test.sh                - Quick functionality test

### Configuration
scripts/env_config.sh        - Configure environment

## File Structure

scripts/          → Automation and management
docs/             → Documentation and guides
tools/            → Python tools (llama-bench, llama-run)
tests/            → Test suite
.venv/            → Python virtual environment
build/            → llama.cpp build artifacts

## Common Tasks

1. Initial Setup
./scripts/setup.sh

2. Start Server
./scripts/start_server.sh

3. Run Tests
./scripts/test_glm.sh

4. View Results
./scripts/benchmark.sh

5. Stop Server
./scripts/stop_server.sh

