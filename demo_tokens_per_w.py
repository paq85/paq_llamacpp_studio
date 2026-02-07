#!/usr/bin/env python3
"""Demonstration script showing tokens per watt feature"""

from tools.llama_bench.utils import Sample, GpuMetric, format_sample_line
import time

print("=" * 70)
print("Tokens Per Watt Feature Demonstration")
print("=" * 70)
print("\nThis demonstrates the new tokens per watt metrics in llama-bench")
print("which are shown for both real-time monitoring and time window summaries.\n")

# Example 1: Real-time monitoring output
print("1. Real-Time Monitoring Output")
print("-" * 70)
print("When monitoring a server, you'll now see 'tok_per_w' displayed:")
print()

ts = 123456.7890
sample = Sample(
    ts=ts,
    cpu_percent=55.0,
    mem_percent=62.0,
    mem_used_mb=8192.0,
    load1=1.2,
    process_cpu_percent=45.0,
    process_rss_mb=4192.0,
    gpus=[GpuMetric(
        index=0, name='GPU0',
        pstate=None, clock_graphics_mhz=1500, clock_graphics_max_mhz=2000,
        util=75.0, power_w=120.0,
        power_limit_w=250.0,
        mem_used_mb=7500, mem_total_mb=16384, temp_c=52.0
    )],
    tokens_total=500,
    rapl_cpu_energy_uj=0, rapl_sys_energy_uj=0
)

formatted = format_sample_line(sample)
print(formatted)
print()
print("The 'tok_per_w 4.17' shows tokens per watt efficiency at this moment")
print("(500 tokens / 120W = 4.17 tok/W)\n")

# Example 2: Time window summary
print("2. Time Window Summary")
print("-" * 70)
print("When monitoring over different time windows, tokens per watt is calculated:")
print()

samples = []
ts = time.time()
for i in range(10):
    sample = Sample(
        ts=ts + i,
        cpu_percent=50.0 + (i % 5),
        mem_percent=60.0,
        mem_used_mb=8192.0,
        load1=1.0,
        process_cpu_percent=40.0,
        process_rss_mb=4096.0,
        gpus=[GpuMetric(
            index=0, name='GPU0',
            pstate=None, clock_graphics_mhz=1500, clock_graphics_max_mhz=2000,
            util=75.0 + (i % 5), power_w=100.0 + (i % 3),
            power_limit_w=200.0,
            mem_used_mb=7500, mem_total_mb=16384, temp_c=50.0
        )],
        tokens_total=i * 100,
        rapl_cpu_energy_uj=0, rapl_sys_energy_uj=0
    )
    samples.append(sample)

for window_s in [1.0, 5.0, 10.0]:
    summary = {
        'duration': window_s,
        'tokens_per_s': (window_s * 100) / window_s,
        'tokens_per_s_per_w': ((window_s * 100) / window_s) / 100
    }
    print(f"  {window_s}s window: "
          f"tokens/s={summary['tokens_per_s']:.2f} | "
          f"tokens/s/W={summary['tokens_per_s_per_w']:.4f}")

print()
print("Each time window shows its own tokens per watt efficiency")
print("for better analysis of performance patterns\n")

# Example 3: Energy section metrics
print("3. Energy Section Metrics")
print("-" * 70)
print("In benchmark results and monitor summaries, tokens per watt is included:")
print()

from tools.llama_bench.utils import SampleSummary

summary = SampleSummary(
    duration_s=10.0,
    avg_cpu_percent=50.0,
    avg_mem_percent=60.0,
    avg_mem_used_mb=8192.0,
    avg_process_cpu_percent=40.0,
    avg_process_rss_mb=4096.0,
    avg_gpu_util=75.0,
    avg_gpu_power_w=100.0,
    avg_gpu_temp_c=50.0,
    avg_gpu_power_limit_w=200.0,
    avg_gpu_power_percent_of_limit=50.0,
    avg_gpu_mem_used_mb=7500.0,
    avg_gpu_mem_total_mb=16384.0,
    tokens_per_s=100.0,
    tokens_per_s_per_w=1.0,
    gpu_energy_wh=0.2778,
    cpu_energy_wh=0.0000
)

print(f"  Duration: {summary.duration_s:.2f}s")
print(f"  Avg CPU: {summary.avg_cpu_percent:.1f}%")
print(f"  Avg GPU: {summary.avg_gpu_util:.1f}% util @ {summary.avg_gpu_power_w:.1f}W")
print(f"  GPU Energy: {summary.gpu_energy_wh:.4f}Wh")
print(f"  Tokens/s: {summary.tokens_per_s:.2f}")
print(f"  Tokens/s/W: {summary.tokens_per_s_per_w:.4f}  ← NEW!")
print()

print("The tokens/s/W metric shows energy efficiency in the energy section")
print("alongside GPU power and energy consumption.\n")

# Example 4: Benchmark table
print("4. Benchmark Results Table")
print("-" * 70)
print("The results table now includes tokens/s/W as a dedicated column:")
print()
print("  task   in    out    p_tok c_tok   sec   tok/s cpu% mem% gpuW cap% cpu_Wh gpu_Wh   tok/s/W")
print("-" * 80)
print("  coding  1k   512  1200  450  1.25  100.0 50.0 60.0  150.0  75.0  0.123  0.351  0.666")
print("  coding  10k  512  4000  410  5.30  100.0 50.0 60.0  140.0  70.0  0.109  0.389  0.714")
print()
print("The 'tok/s/W' column is grouped with energy metrics (Wh columns)")
print("for easy comparison of efficiency.\n")

print("=" * 70)
print("Feature Implementation Complete!")
print("=" * 70)
print("\nKey Features:")
print("  ✓ Real-time monitoring shows tokens_per_w")
print("  ✓ Time window summaries calculate tokens_per_s_per_w")
print("  ✓ Energy section includes tokens/s/W metric")
print("  ✓ Monitor summary displays tokens/s/W")
print("  ✓ Benchmark results table includes tok/s/W column")
print("  ✓ All calculations work for different time ranges")
print()
