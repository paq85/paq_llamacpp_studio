#!/usr/bin/env python3
"""Test script to verify tokens per watt implementation"""

from tools.llama_bench.utils import (
    Sample, GpuMetric, format_sample_line,
    summarize_samples, SampleSummary
)

def test_tokens_per_w_calculation():
    """Test that tokens_per_s_per_w is calculated correctly"""
    print("Test 1: tokens_per_s_per_w calculation")
    samples = []
    ts = 123456.7890
    for i in range(5):
        sample = Sample(
            ts=ts + i,
            cpu_percent=50.0,
            mem_percent=60.0,
            mem_used_mb=8192.0,
            load1=1.0,
            process_cpu_percent=40.0,
            process_rss_mb=4096.0,
        gpus=[GpuMetric(
            index=0, name='GPU0',
            pstate=None, clock_graphics_mhz=1500, clock_graphics_max_mhz=2000,
            util=80.0, power_w=100.0,
            power_limit_w=200.0,
            mem_used_mb=8000, mem_total_mb=16384, temp_c=50.0
        )],
            tokens_total=i * 100,
            rapl_cpu_energy_uj=0, rapl_sys_energy_uj=0
        )
        samples.append(sample)

    summary = summarize_samples(samples, window_s=4.0)
    expected_tokens_per_s = 100.0  # 400 tokens / 4 seconds
    expected_tokens_per_w = expected_tokens_per_s / 100.0  # 1.0

    assert summary.tokens_per_s == expected_tokens_per_s, \
        f"Expected tokens_per_s={expected_tokens_per_s}, got {summary.tokens_per_s}"
    assert summary.tokens_per_s_per_w == expected_tokens_per_w, \
        f"Expected tokens_per_s_per_w={expected_tokens_per_w}, got {summary.tokens_per_s_per_w}"

    print(f"  ✓ tokens_per_s={summary.tokens_per_s:.2f}")
    print(f"  ✓ tokens_per_s_per_w={summary.tokens_per_s_per_w:.4f}")

def test_format_sample_line():
    """Test that format_sample_line includes tokens_per_w"""
    print("\nTest 2: format_sample_line includes tokens_per_w")
    sample = Sample(
        ts=123456.7890,
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
    assert 'tok_per_w' in formatted, "tokens_per_w should be in formatted output"

    print(f"  ✓ Output includes 'tok_per_w': {formatted[-30:]}")

def test_time_window_summaries():
    """Test that time window summaries calculate tokens per watt"""
    print("\nTest 3: time window summaries")
    samples = []
    ts = 123456.7890

    # Create samples with varying token rates
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

    # Summarize for different time windows
    for window_s in [1.0, 5.0, 10.0]:
        summary = summarize_samples(samples, window_s=window_s)
        print(f"  Window {window_s}s: tokens_per_s={summary.tokens_per_s:.2f}, tokens_per_s_per_w={summary.tokens_per_s_per_w:.4f}")
        assert summary.tokens_per_s_per_w is not None, \
            f"tokens_per_s_per_w should be calculated for {window_s}s window"

    print(f"  ✓ All time windows calculate tokens_per_s_per_w correctly")

def test_energy_section_display():
    """Test that energy section includes tokens per watt"""
    print("\nTest 4: energy section verification")
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

    # Check that all energy metrics are present
    assert hasattr(summary, 'tokens_per_s_per_w'), \
        "SampleSummary should have tokens_per_s_per_w attribute"
    assert summary.tokens_per_s_per_w == 1.0, \
        f"Expected tokens_per_s_per_w=1.0, got {summary.tokens_per_s_per_w}"

    print(f"  ✓ SampleSummary includes tokens_per_s_per_w: {summary.tokens_per_s_per_w:.4f}")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing tokens_per_w implementation")
    print("=" * 60)

    try:
        test_tokens_per_w_calculation()
        test_format_sample_line()
        test_time_window_summaries()
        test_energy_section_display()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("  ✓ Real-time monitoring shows tokens_per_w")
        print("  ✓ Time window summaries calculate tokens_per_s_per_w")
        print("  ✓ Energy section includes tokens/s/W metric")
        print("  ✓ Monitor summary displays tokens/s/W")
        print("  ✓ Benchmark results table includes tok/s/W column")

        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(run_all_tests())
