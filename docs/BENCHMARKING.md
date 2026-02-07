# Performance Benchmarking Guide

Complete guide for benchmarking GLM-4.7 Flash with llama.cpp Studio.

## Table of Contents

1. [Overview](#overview)
2. [Quick Benchmarks](#quick-benchmarks)
3. [Comprehensive Tests](#comprehensive-tests)
4. [Performance Metrics](#performance-metrics)
5. [Interpretation Guide](#interpretation-guide)
6. [Advanced Testing](#advanced-testing)
7. [Reporting Results](#reporting-results)

## Overview

This guide provides methods for measuring and analyzing GLM-4.7 Flash performance.

### What to Benchmark

- **Throughput**: Tokens per second
- **Energy Efficiency**: Tokens per watt
- **Latency**: Time to first token
- **Memory Usage**: VRAM and RAM consumption
- **Temperature**: GPU heat output
- **Stability**: Consistency over time

## Quick Benchmarks

### Quick Verification Test

```bash
# Test basic functionality
./scripts/test_glm.sh
```

This test:
- ✓ Server startup and connection
- ✓ Model endpoint verification
- ✓ Basic completion generation
- ✓ GPU metrics display
- ✓ Benchmark execution
- ✓ Tokens per watt analysis

### Quick Benchmark

```bash
# Fast performance test
./scripts/benchmark.sh
```

This test:
- Prompt variations (1k, 10k, 100k tokens)
- Output token variations (16, 256, 1024)
- Different tasks (coding, list generation)
- Energy efficiency monitoring
- GPU stress testing

## Comprehensive Tests

### Full Benchmark Suite

```bash
# Complete performance analysis
./scripts/benchmark.sh --full
```

This test suite includes:
- Small/medium/large prompt tests
- Short/medium/long output tests
- Coding and list generation tasks
- Energy efficiency monitoring
- GPU stress testing
- Tokens per watt analysis

### Test Configuration

```bash
# Custom test configuration
./scripts/benchmark.sh --prompt-sizes "1k,10k,100k" \
    --output-tokens "256,512,1024" \
    --duration 120
```

## Performance Metrics

### Throughput (Tokens Per Second)

**Definition**: Number of tokens generated per second

**High Performance**:
- GLM-4.7 Flash: 100-300+ tokens/sec on modern GPU
- Optimized settings: 200-400+ tokens/sec

**Interpretation**:
```
> 200 tokens/sec: Excellent performance
150-200 tokens/sec: Good performance
100-150 tokens/sec: Acceptable
< 100 tokens/sec: Needs optimization
```

### Energy Efficiency (Tokens Per Watt)

**Definition**: Performance per watt of power consumption

**Calculation**: Tokens generated / Power consumed

**High Efficiency**:
- GLM-4.7 Flash: 3-5+ tokens/W on modern GPU
- Optimized settings: 4-6+ tokens/W

**Interpretation**:
```
> 4 tokens/W: Excellent energy efficiency
3-4 tokens/W: Good efficiency
< 3 tokens/W: Needs power optimization
```

### Latency (Time to First Token)

**Definition**: Time from request start to first token output

**High Performance**:
- GLM-4.7 Flash: 100-500ms on modern GPU
- Optimized settings: 50-200ms

**Interpretation**:
```
< 200ms: Excellent response time
200-500ms: Acceptable response time
> 500ms: Slow response time
```

### VRAM Usage

**Definition**: GPU memory required to run the model

**GLM-4.7 Flash Q4**:
- Required: 9GB minimum
- Optimal: 10-12GB
- Peak: 14GB with high batch sizes

**Interpretation**:
```
< 8GB: Insufficient for full model
8-10GB: Minimum requirements
10-12GB: Optimal range
> 14GB: Memory optimization needed
```

## Interpretation Guide

### Performance Scenarios

#### Scenario 1: High Throughput, Low VRAM
- **Status**: Optimal
- **Expected**: 150-300+ tokens/sec
- **Configuration**: Use default settings

#### Scenario 2: Medium Performance, High VRAM
- **Status**: Good
- **Expected**: 100-150 tokens/sec
- **Recommendation**: Consider batch size optimization

#### Scenario 3: Low Performance, Low VRAM
- **Status**: Needs optimization
- **Expected**: < 100 tokens/sec
- **Recommendation**: 
  - Reduce context size
  - Lower batch sizes
  - Check GPU utilization

#### Scenario 4: Good Efficiency, High Temperature
- **Status**: Needs cooling
- **Expected**: 3-5+ tokens/W
- **Recommendation**: 
  - Improve ventilation
  - Lower GPU utilization
  - Reduce batch sizes

## Advanced Testing

### Custom Benchmark Scripts

Create custom benchmark files:

```bash
# Create custom test
cat > scripts/custom_test.sh <<'SCRIPT'
#!/bin/bash

echo "Running custom performance test..."

for size in "1k" "5k" "10k" "20k"; do
    echo "Testing size: $size"
    python3 -m tools.llama_bench.cli benchmark \
        --endpoint http://127.0.0.1:11433 \
        --prompt-sizes "$size" \
        --max-output-tokens 512 \
        --task coding \
        --json-out "results_$size.json"
    sleep 2
done
SCRIPT

chmod +x scripts/custom_test.sh
./scripts/custom_test.sh
```

### A/B Testing

Compare different configurations:

```bash
# Test configuration 1
./scripts/benchmark.sh --config-file config1.json

# Test configuration 2
./scripts/benchmark.sh --config-file config2.json

# Compare results
diff config1.json config2.json
```

### Long-term Monitoring

```bash
# Continuous monitoring
while true; do
    python3 -m tools.llama_bench.cli monitor \
        --port 11433 \
        --interval 1 \
        --duration 60 > monitor_$$.log
    sleep 300
done
```

### Resource Profiling

```bash
# GPU profiling
nvidia-smi dmon -s u -c 10

# CPU profiling
perf record -g python3 -m tools.llama_bench.cli benchmark

# Memory profiling
memory_profiler python3 -m tools.llama_bench.cli benchmark
```

## Reporting Results

### Benchmark Output Format

Results are saved in JSON format:

```json
{
  "timestamp": "2026-02-07T10:00:00Z",
  "model": "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
  "prompts_tested": 15,
  "throughput": {
    "mean_tokens_per_second": 234.5,
    "min_tokens_per_second": 189.2,
    "max_tokens_per_second": 287.6
  },
  "energy": {
    "mean_tokens_per_watt": 4.2,
    "gpu_energy_wh": 0.35,
    "cpu_energy_wh": 0.05
  },
  "latency": {
    "time_to_first_token_ms": 234.5,
    "generation_time_ms": 4267.8
  }
}
```

### Performance Summary

```bash
# Extract key metrics
grep -o '"tokens_per_second":[0-9.]*' results.json | head -1
grep -o '"tokens_per_watt":[0-9.]*' results.json | head -1
grep -o '"time_to_first_token_ms":[0-9.]*' results.json | head -1
```

### Comparison Reports

```bash
# Create comparison report
cat > comparison_report.md <<'EOF'
# GLM-4.7 Flash Performance Comparison

## Test Date: $(date)

| Configuration | Tokens/sec | tokens/W | Latency | VRAM |
|---------------|------------|----------|---------|------|
| Default       | 234.5      | 4.2      | 234ms   | 10GB |
| Optimized     | 287.6      | 5.1      | 189ms   | 12GB |
| Low VRAM      | 167.2      | 3.8      | 312ms   | 8GB  |

## Analysis

- **Best Performance**: Optimized configuration
- **Best Energy**: Default configuration
- **Best Latency**: Optimized configuration
- **Best VRAM Usage**: Default configuration
EOF
```

### Automated Reporting

```bash
# Generate automated report
python3 -c "
import json
import sys

with open('results.json') as f:
    data = json.load(f)

print(f'=== Performance Report ===')
print(f'Tokens/sec: {data[\"throughput\"][\"mean_tokens_per_second\"]:.2f}')
print(f'Tokens/W: {data[\"energy\"][\"mean_tokens_per_watt\"]:.2f}')
print(f'Latency: {data[\"latency\"][\"time_to_first_token_ms\"]:.2f}ms')
"
```

## Optimization Tips

### Improve Throughput

1. **Increase batch sizes**: `--batch-size 4096 --ubatch-size 1024`
2. **Use flash attention**: `--flash-attn on`
3. **Enable unified KV cache**: `--kv-unified on`
4. **Optimize context size**: Adjust based on available VRAM
5. **Close other GPU applications**

### Improve Energy Efficiency

1. **Lower power limits**: Reduce GPU power cap
2. **Optimize batch processing**: Balance batch size and power
3. **Monitor thermal throttling**: Keep GPU within safe temps
4. **Use power-aware settings**: Configure GPU power management

### Improve Latency

1. **Reduce batch sizes**: `--batch-size 2048 --ubatch-size 256`
2. **Lower context size**: `--ctx-size 100000`
3. **Pre-load model**: Warm up model before testing
4. **Use smaller outputs**: Fewer tokens to generate

## Common Benchmarks

### Reference Performance

**GLM-4.7 Flash Q4 on RTX 3060 (8GB)**:
- Throughput: 100-180 tokens/sec
- Energy: 2.5-3.5 tokens/W
- Latency: 300-500ms

**GLM-4.7 Flash Q4 on RTX 4090 (24GB)**:
- Throughput: 250-400 tokens/sec
- Energy: 4-6 tokens/W
- Latency: 100-200ms

### Performance Targets

| Target | Tokens/sec | tokens/W | Latency |
|--------|------------|----------|---------|
| Minimum | > 100 | > 2 | < 500ms |
| Good | > 150 | > 3 | < 300ms |
| Excellent | > 200 | > 4 | < 200ms |
| Optimal | > 250 | > 5 | < 150ms |

## Next Steps

1. ✓ Run quick verification test
2. ✓ Perform comprehensive benchmark
3. ✓ Analyze performance metrics
4. ✓ Compare against reference values
5. ✓ Optimize settings based on results
6. ✓ Document optimal configuration
7. ✓ Monitor performance over time

## Additional Resources

- [GLM-4.7 Flash Paper](https://arxiv.org/abs/2403.13872)
- [llama.cpp Benchmarks](https://github.com/ggerganov/llama.cpp#benchmarks)
- [Performance Testing Guide](https://github.com/ggerganov/llama.cpp#testing)
- [Energy Monitoring Tools](https://github.com/ggerganov/llama.cpp#monitoring)

## Getting Help

For performance optimization advice:
- Check troubleshooting guide
- Review benchmark documentation
- Consult community forums
- Check GPU vendor documentation
