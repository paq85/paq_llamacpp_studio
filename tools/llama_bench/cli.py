import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from . import autotune
from . import benchmark
from . import utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-bench",
        description="Run llama.cpp server, monitor resources, and benchmark prompts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", help="Run llama.cpp server directly")
    server.add_argument("--llama-server", help="Path to llama-server binary (otherwise autodetect)")
    server.add_argument("--llama-cpp-dir", help="llama.cpp directory (LLAMA_CPP_DIR env var)")
    server.add_argument("--model", help="Path to .gguf model (MODEL env var)")
    server.add_argument("--model-dir", help="Search for .gguf models in this directory")
    server.add_argument("--model-alias", help="Model alias (defaults to model filename)")
    server.add_argument("--host", help="Bind host (passed to --host)")
    server.add_argument("--port", type=int, default=11433, help="Server port")
    server.add_argument("--min-vram-free-mb", type=int, default=8192, help="Minimum free VRAM required")
    server.add_argument("--skip-vram-check", action="store_true", help="Skip VRAM availability check")
    server.add_argument(
        "--allow-multi-server",
        action="store_true",
        help="Allow starting even if another llama-server is running",
    )
    server.add_argument("--ctx-size", type=int, default=200000, help="Context size")
    server.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    server.add_argument("--ubatch-size", type=int, default=1024, help="Micro-batch size")
    server.add_argument("--cache-type-k", default="q8_0", help="KV cache type for K")
    server.add_argument("--cache-type-v", default="q8_0", help="KV cache type for V")
    server.add_argument("--fit", choices=["on", "off"], default="on", help="Enable context fitting")
    server.add_argument("--flash-attn", choices=["on", "off"], default="on", help="Enable flash attention")
    server.add_argument("--kv-unified", choices=["on", "off"], default="on", help="Enable unified KV")
    server.add_argument("--jinja", choices=["on", "off"], default="on", help="Enable Jinja chat templates")
    server.add_argument("--chat-template-kwargs", help="JSON for --chat-template-kwargs")
    server.add_argument("--enable-thinking", choices=["true", "false"], default="true", help="Enable thinking flag")
    server.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    server.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    server.add_argument("--min-p", type=float, default=0.01, help="Min-p sampling")
    server.add_argument("--monitor", action="store_true", help="Monitor resources while server runs")
    server.add_argument("--interval", type=float, default=1.0, help="Monitor interval (seconds)")
    server.add_argument("--metrics-url", help="Prometheus metrics endpoint (default: http://localhost:PORT/metrics)")
    server.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to llama-server (use after --)",
    )

    monitor = subparsers.add_parser("monitor", help="Monitor system/GPU resources")
    monitor.add_argument("--interval", type=float, default=1.0, help="Sampling interval (seconds)")
    monitor.add_argument("--duration", type=float, help="Stop after duration (seconds)")
    monitor.add_argument("--pid", type=int, help="Process ID to track")
    monitor.add_argument("--port", type=int, help="Find PID by server port")
    monitor.add_argument("--metrics-url", help="Prometheus metrics endpoint (default: http://localhost:PORT/metrics)")
    monitor.add_argument("--json-out", help="Write samples summary to JSON file")

    bench = subparsers.add_parser("benchmark", help="Run prompt-length benchmark")
    bench.add_argument("--endpoint", default="http://127.0.0.1:11433", help="Server base URL")
    bench.add_argument("--mode", choices=["chat", "completion"], default="chat", help="API mode")
    bench.add_argument(
        "--task",
        default="coding",
        help="Benchmark task prompt (coding or list). Comma-separated to run multiple.",
    )
    bench.add_argument(
        "--prompt-sizes",
        default="1k,10k,100k",
        help="Prompt sizes in tokens (comma separated, supports k/m)",
    )
    bench.add_argument("--max-output-tokens", type=int, default=1024, help="Max output tokens")
    bench.add_argument(
        "--output-tokens",
        help="Comma-separated output token counts to test (overrides --max-output-tokens)",
    )
    bench.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    bench.add_argument("--model", help="Model name for OpenAI-compatible API")
    bench.add_argument("--extra", help="Extra JSON fields to include in requests")
    bench.add_argument("--interval", type=float, default=1.0, help="Sampling interval (seconds)")
    bench.add_argument("--pid", type=int, help="Process ID to track")
    bench.add_argument("--port", type=int, help="Find PID by server port")
    bench.add_argument("--metrics-url", help="Prometheus metrics endpoint (default: http://localhost:PORT/metrics)")
    bench.add_argument("--json-out", help="Write benchmark results to JSON file")
    bench.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Wait up to this long for server readiness (seconds)",
    )
    bench.add_argument(
        "--retry-503",
        type=float,
        default=60.0,
        help="Retry for this long if server returns HTTP 503 (seconds)",
    )
    bench.add_argument(
        "--progress",
        choices=["on", "off"],
        default="on",
        help="Print periodic progress while each request runs",
    )
    bench.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="How often to print progress (seconds)",
    )

    # Convenience: ensure a server is running, then benchmark it.
    # Reuses an existing llama-server when possible; otherwise starts one.
    eb = subparsers.add_parser(
        "bench",
        help="Ensure llama-server is running, then run benchmark",
    )
    eb.add_argument("--llama-server", help="Path to llama-server binary (otherwise autodetect)")
    eb.add_argument("--llama-cpp-dir", help="llama.cpp directory (LLAMA_CPP_DIR env var)")
    eb.add_argument("--model", help="Path to .gguf model (MODEL env var)")
    eb.add_argument("--model-dir", help="Search for .gguf models in this directory")
    eb.add_argument("--model-alias", help="Model alias (defaults to model filename)")
    eb.add_argument("--host", default="127.0.0.1", help="Bind host")
    eb.add_argument("--port", type=int, default=11433, help="Server port")
    eb.add_argument("--min-vram-free-mb", type=int, default=8192, help="Minimum free VRAM required")
    eb.add_argument("--skip-vram-check", action="store_true", help="Skip VRAM availability check")
    eb.add_argument(
        "--allow-multi-server",
        action="store_true",
        help="Allow starting even if another llama-server is running",
    )
    eb.add_argument("--startup-timeout", type=float, default=180.0, help="Server readiness timeout (seconds)")
    eb.add_argument("--keep-server", action="store_true", help="Keep server running after benchmark")

    # server settings
    eb.add_argument("--ctx-size", type=int, default=200000, help="Context size")
    eb.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    eb.add_argument("--ubatch-size", type=int, default=1024, help="Micro-batch size")
    eb.add_argument("--cache-type-k", default="q8_0", help="KV cache type for K")
    eb.add_argument("--cache-type-v", default="q8_0", help="KV cache type for V")
    eb.add_argument("--fit", choices=["on", "off"], default="on", help="Enable context fitting")
    eb.add_argument("--flash-attn", choices=["on", "off"], default="on", help="Enable flash attention")
    eb.add_argument("--kv-unified", choices=["on", "off"], default="on", help="Enable unified KV")
    eb.add_argument("--jinja", choices=["on", "off"], default="on", help="Enable Jinja chat templates")
    eb.add_argument("--chat-template-kwargs", help="JSON for --chat-template-kwargs")
    eb.add_argument("--enable-thinking", choices=["true", "false"], default="true", help="Enable thinking flag")
    eb.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    eb.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    eb.add_argument("--min-p", type=float, default=0.01, help="Min-p sampling")
    eb.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to llama-server (use after --)",
    )

    # benchmark settings
    eb.add_argument("--mode", choices=["chat", "completion"], default="chat", help="API mode")
    eb.add_argument(
        "--task",
        default="coding",
        help="Benchmark task prompt (coding or list). Comma-separated to run multiple.",
    )
    eb.add_argument(
        "--prompt-sizes",
        default="1k,10k,100k",
        help="Prompt sizes in tokens (comma separated, supports k/m)",
    )
    eb.add_argument("--max-output-tokens", type=int, default=1024, help="Max output tokens")
    eb.add_argument(
        "--output-tokens",
        help="Comma-separated output token counts to test (overrides --max-output-tokens)",
    )
    eb.add_argument("--temperature", type=float, default=0.0, help="Benchmark sampling temperature")
    eb.add_argument("--api-model", help="Model name for OpenAI-compatible API")
    eb.add_argument("--extra", help="Extra JSON fields to include in requests")
    eb.add_argument("--interval", type=float, default=1.0, help="Sampling interval (seconds)")
    eb.add_argument("--metrics-url", help="Prometheus metrics endpoint (default: http://HOST:PORT/metrics)")
    eb.add_argument("--json-out", help="Write benchmark results to JSON file")
    eb.add_argument(
        "--progress",
        choices=["on", "off"],
        default="on",
        help="Print periodic progress while each request runs",
    )
    eb.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="How often to print progress (seconds)",
    )
    eb.add_argument(
        "--retry-503",
        type=float,
        default=60.0,
        help="Retry for this long if server returns HTTP 503 (seconds)",
    )

    tune = subparsers.add_parser("autotune", help="Auto-tune llama.cpp server settings")
    tune.add_argument("--llama-server", help="Path to llama-server binary (otherwise autodetect)")
    tune.add_argument("--llama-cpp-dir", help="llama.cpp directory (LLAMA_CPP_DIR env var)")
    tune.add_argument("--model", help="Path to .gguf model (MODEL env var)")
    tune.add_argument("--model-dir", help="Search for .gguf models in this directory")
    tune.add_argument("--host", default="127.0.0.1", help="Bind host")
    tune.add_argument("--ctx-size", type=int, default=200000, help="Context size")
    tune.add_argument("--batch-sizes", default="1024,2048,4096", help="Comma-separated batch sizes")
    tune.add_argument("--ubatch-sizes", default="256,512,1024", help="Comma-separated micro-batch sizes")
    tune.add_argument("--duration", type=float, default=600.0, help="Max autotune duration (seconds)")
    tune.add_argument("--prompt-tokens", type=int, default=1024, help="Prompt tokens used per test")
    tune.add_argument("--max-output-tokens", type=int, default=256, help="Max output tokens per test")
    tune.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    tune.add_argument(
        "--objective",
        choices=["throughput", "efficiency"],
        default="throughput",
        help="Optimization objective",
    )
    tune.add_argument("--interval", type=float, default=1.0, help="Sampling interval (seconds)")
    tune.add_argument("--startup-timeout", type=float, default=30.0, help="Server startup timeout (seconds)")
    tune.add_argument("--min-vram-free-mb", type=int, default=8192, help="Minimum free VRAM required")
    tune.add_argument("--skip-vram-check", action="store_true", help="Skip VRAM availability check")
    tune.add_argument(
        "--allow-multi-server",
        action="store_true",
        help="Allow starting even if another llama-server is running",
    )
    tune.add_argument("--cache-type-k", default="q8_0", help="KV cache type for K")
    tune.add_argument("--cache-type-v", default="q8_0", help="KV cache type for V")
    tune.add_argument("--fit", choices=["on", "off"], default="on", help="Enable context fitting")
    tune.add_argument("--flash-attn", choices=["on", "off"], default="on", help="Enable flash attention")
    tune.add_argument("--kv-unified", choices=["on", "off"], default="on", help="Enable unified KV")
    tune.add_argument("--jinja", choices=["on", "off"], default="on", help="Enable Jinja chat templates")
    tune.add_argument("--chat-template-kwargs", help="JSON for --chat-template-kwargs")
    tune.add_argument("--enable-thinking", choices=["true", "false"], default="true", help="Enable thinking flag")
    tune.add_argument("--extra-request", help="Extra JSON fields to include in requests")
    tune.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "tuned.json"),
        help="Path to save best settings",
    )
    tune.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed to llama-server (use after --)")

    return parser


def resolve_metrics_url(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "metrics_url", None):
        return args.metrics_url
    port = getattr(args, "port", None)
    if port:
        return f"http://127.0.0.1:{port}/metrics"
    return None


def infer_port_from_cmdline(cmdline: str) -> Optional[int]:
    # Best-effort parse of llama-server cmdline string.
    # utils.find_llama_server_processes() returns a joined string without quoting.
    match = re.search(r"\s--port\s+(\d+)(?:\s|$)", f" {cmdline} ")
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def parse_int_list_csv(value: Optional[str]) -> List[int]:
    if not value:
        return []
    out: List[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def format_results_table(results: List[benchmark.BenchmarkResult]) -> str:
    cols = [
        ("task", lambda r: str(r.task)),
        ("in", lambda r: str(r.prompt_target_tokens)),
        ("out", lambda r: str(r.max_output_tokens)),
        ("p_tok", lambda r: "n/a" if r.prompt_tokens is None else str(r.prompt_tokens)),
        ("c_tok", lambda r: "n/a" if r.completion_tokens is None else str(r.completion_tokens)),
        ("sec", lambda r: f"{r.duration_s:.2f}"),
        ("tok/s", lambda r: "n/a" if r.tokens_per_s is None else f"{r.tokens_per_s:.1f}"),
        ("gpuW", lambda r: "n/a" if r.avg_gpu_power_w is None else f"{r.avg_gpu_power_w:.1f}"),
        ("cap%", lambda r: "n/a" if r.avg_gpu_power_percent_of_limit is None else f"{r.avg_gpu_power_percent_of_limit:.1f}"),
        ("Wh", lambda r: "n/a" if r.gpu_energy_wh is None else f"{r.gpu_energy_wh:.3f}"),
        ("tok/s/W", lambda r: "n/a" if r.tokens_per_s_per_w is None else f"{r.tokens_per_s_per_w:.3f}"),
    ]

    rows: List[List[str]] = [[name for name, _ in cols]]
    for r in results:
        rows.append([fn(r) for _, fn in cols])

    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
    lines: List[str] = []
    lines.append(" ".join(rows[0][i].rjust(widths[i]) for i in range(len(cols))))
    lines.append(" ".join(("-" * widths[i]) for i in range(len(cols))))
    for row in rows[1:]:
        lines.append(" ".join(row[i].rjust(widths[i]) for i in range(len(cols))))
    return "\n".join(lines)


def find_llama_server_on_port(port: int) -> Optional[int]:
    pid = utils.find_pid_by_port(port)
    if not pid:
        return None
    for p, _cmd in utils.find_llama_server_processes():
        if p == pid:
            return pid
    return None


def wait_for_server_ready(endpoint: str, timeout_s: float = 180.0) -> bool:
    """Wait for OpenAI-compatible endpoints to respond.

    We try cheap requests first because llama-server can bind the port before it
    is ready to serve completions.
    """
    utils.require(utils.requests, "requests")
    deadline = time.time() + timeout_s
    url_models = endpoint.rstrip("/") + "/v1/models"
    url_metrics = endpoint.rstrip("/") + "/metrics"
    while time.time() < deadline:
        try:
            r = utils.requests.get(url_models, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        try:
            r = utils.requests.get(url_metrics, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def cmd_server(args: argparse.Namespace) -> int:
    if not args.allow_multi_server:
        existing = utils.find_llama_server_processes()
        if existing:
            # Prefer reusing the server bound to the requested port.
            selected_pid: Optional[int] = None
            selected_cmd: Optional[str] = None
            selected_port: Optional[int] = None

            if args.port:
                pid_by_port = utils.find_pid_by_port(args.port)
                if pid_by_port:
                    for pid, cmdline in existing:
                        if pid == pid_by_port:
                            selected_pid = pid
                            selected_cmd = cmdline
                            selected_port = args.port
                            break

            if selected_pid is None and len(existing) == 1:
                selected_pid, selected_cmd = existing[0]
                selected_port = infer_port_from_cmdline(selected_cmd or "")

            if selected_pid is None:
                print(
                    "Detected running llama-server processes. Refusing to start another one. "
                    "Specify --port to reuse a specific instance, or pass --allow-multi-server to override:",
                    file=sys.stderr,
                )
                for pid, cmdline in existing:
                    line = cmdline or "llama-server"
                    print(f"  pid {pid}: {line}", file=sys.stderr)
                return 1

            print(f"Reusing existing llama-server pid {selected_pid}")
            if selected_cmd:
                print(f"  cmd: {selected_cmd}")
            if selected_port:
                print(f"  port: {selected_port}")

            # If monitoring requested, attach to the existing process and do not start a new server.
            if args.monitor:
                args.pid = selected_pid
                if selected_port:
                    args.port = selected_port
                return cmd_monitor(args)

            return 0

    ok, reason = utils.check_port_available(args.port)
    if not ok:
        print(reason, file=sys.stderr)
        return 1

    if not args.skip_vram_check:
        ok, reason = utils.check_vram_available(float(args.min_vram_free_mb))
        if not ok:
            print(reason, file=sys.stderr)
            return 1

    llama_cpp_dir = args.llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    llama_server = args.llama_server or utils.find_llama_server(llama_cpp_dir)
    if not llama_server:
        print(
            "llama-server not found. Provide --llama-server or --llama-cpp-dir, "
            "or ensure llama-server is on PATH.",
            file=sys.stderr,
        )
        return 1

    model_path = utils.find_model_path(
        args.model,
        args.model_dir,
        preferred_files=utils.DEFAULT_MODEL_FILES,
        preferred_substrings=utils.DEFAULT_MODEL_SUBSTRINGS,
    )
    if not model_path:
        candidates = utils.list_models(args.model_dir)
        if not candidates:
            print(
                "Model not found. Provide --model or set MODEL env var, or place a .gguf in ~/models.",
                file=sys.stderr,
            )
        else:
            print("Multiple models found. Choose one with --model:", file=sys.stderr)
            for item in candidates:
                print(f"  {item}", file=sys.stderr)
        return 1

    model_alias = args.model_alias or utils.default_model_alias(model_path)

    cmd = [
        llama_server,
        "--model",
        model_path,
        "--alias",
        model_alias,
        "--port",
        str(args.port),
        "--fit",
        args.fit,
        "--temp",
        str(args.temp),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--flash-attn",
        args.flash_attn,
    ]
    if args.host:
        cmd.extend(["--host", args.host])
    if args.kv_unified == "on":
        cmd.append("--kv-unified")
    if args.jinja == "on":
        cmd.append("--jinja")
        if args.chat_template_kwargs:
            cmd.extend(["--chat-template-kwargs", args.chat_template_kwargs])
        elif args.enable_thinking:
            cmd.extend(["--chat-template-kwargs", f'{{\"enable_thinking\": {args.enable_thinking}}}'])

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    proc = subprocess.Popen(cmd)

    if not args.monitor:
        return proc.wait()

    metrics_url = resolve_metrics_url(args)
    sampler = utils.MetricsSampler(interval=args.interval, pid=proc.pid, metrics_url=metrics_url)
    sampler.start()

    last_idx = 0
    print("Monitoring resources... press Ctrl+C to stop.")
    try:
        while proc.poll() is None:
            time.sleep(args.interval)
            if len(sampler.samples) > last_idx:
                sample = sampler.samples[-1]
                print(utils.format_sample_line(sample))
                last_idx = len(sampler.samples)
    except KeyboardInterrupt:
        print("Stopping server...")
        proc.send_signal(signal.SIGINT)
    finally:
        sampler.stop()

    return proc.wait()


def cmd_monitor(args: argparse.Namespace) -> int:
    pid = args.pid
    if not pid and args.port:
        pid = utils.find_pid_by_port(args.port)
    if args.port and not pid:
        print(f"No process listening on port {args.port}", file=sys.stderr)

    metrics_url = resolve_metrics_url(args)
    sampler = utils.MetricsSampler(interval=args.interval, pid=pid, metrics_url=metrics_url)
    sampler.start()
    print("Monitoring resources... press Ctrl+C to stop.")

    start = time.time()
    last_idx = 0
    try:
        while True:
            time.sleep(args.interval)
            if len(sampler.samples) > last_idx:
                sample = sampler.samples[-1]
                print(utils.format_sample_line(sample))
                last_idx = len(sampler.samples)
            if args.duration and (time.time() - start) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()

    summary = sampler.summarize()
    print("Summary:")
    print(f"  duration: {utils.human_duration(summary.duration_s)}")
    print(f"  avg cpu: {utils.format_percent(summary.avg_cpu_percent)}")
    print(f"  avg mem: {utils.format_percent(summary.avg_mem_percent)}")
    print(f"  avg gpu util: {utils.format_percent(summary.avg_gpu_util)}")
    print(f"  avg gpu power: {utils.format_number(summary.avg_gpu_power_w, 'W', precision=1)}")
    if summary.tokens_per_s is not None:
        print(f"  tokens/s: {summary.tokens_per_s:.2f}")

    if args.json_out:
        payload = {
            "summary": {
                "duration_s": summary.duration_s,
                "avg_cpu_percent": summary.avg_cpu_percent,
                "avg_mem_percent": summary.avg_mem_percent,
                "avg_gpu_util": summary.avg_gpu_util,
                "avg_gpu_power_w": summary.avg_gpu_power_w,
                "tokens_per_s": summary.tokens_per_s,
            },
            "samples": [sample.__dict__ for sample in sampler.samples],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    prompt_sizes = benchmark.parse_size_list(args.prompt_sizes)
    tasks = [t.strip() for t in str(getattr(args, "task", "coding")).split(",") if t.strip()]
    if not tasks:
        tasks = ["coding"]
    output_tokens_list = parse_int_list_csv(getattr(args, "output_tokens", None))
    if not output_tokens_list:
        output_tokens_list = [int(args.max_output_tokens)]
    extra: Optional[Dict[str, Any]] = None
    if args.extra:
        extra = json.loads(args.extra)

    pid = args.pid
    if not pid and args.port:
        pid = utils.find_pid_by_port(args.port)
    metrics_url = resolve_metrics_url(args)

    # Wait for server readiness (llama-server can bind before it can serve).
    if getattr(args, "startup_timeout", None):
        if not wait_for_server_ready(args.endpoint, timeout_s=float(args.startup_timeout)):
            print(
                f"Warning: server at {args.endpoint} did not report ready endpoints within {args.startup_timeout:.0f}s; continuing...",
                file=sys.stderr,
            )

    results = []
    for task in tasks:
        for size in prompt_sizes:
            for out_tokens in output_tokens_list:
                print(f"Starting benchmark: task={task} in={size} out={out_tokens}")

                t0 = time.time()
                progress_state = {"last": 0.0}

                def on_sample(sample: utils.Sample) -> None:
                    if args.progress != "on":
                        return
                    now = time.time()
                    if (now - float(progress_state["last"])) < float(args.progress_interval):
                        return
                    progress_state["last"] = now
                    elapsed = now - t0
                    sys.stdout.write(
                        f"[progress] task={task} in={size} out={out_tokens} elapsed={elapsed:.1f}s {utils.format_sample_line(sample)}\n"
                    )
                    sys.stdout.flush()

                deadline = time.time() + float(getattr(args, "retry_503", 0.0) or 0.0)
                attempt = 0
                while True:
                    attempt += 1
                    sampler = utils.MetricsSampler(
                        interval=args.interval,
                        pid=pid,
                        metrics_url=metrics_url,
                    )
                    sampler.on_sample = on_sample  # type: ignore[attr-defined]
                    try:
                        result = benchmark.run_single_benchmark(
                            endpoint=args.endpoint,
                            task=task,
                            prompt_tokens_target=size,
                            max_output_tokens=out_tokens,
                            temperature=args.temperature,
                            mode=args.mode,
                            model=args.model,
                            extra=extra,
                            sampler=sampler,
                        )
                        break
                    except Exception as exc:
                        code = getattr(getattr(exc, "response", None), "status_code", None)
                        if code == 503 and time.time() < deadline:
                            print(f"Server returned 503 (attempt {attempt}); retrying...", file=sys.stderr)
                            time.sleep(1.0)
                            continue
                        raise
                results.append(result)

    for result in results:
        print("-")
        print(
            f"Scenario: task={result.task} in={result.prompt_target_tokens} out={result.max_output_tokens} "
            f"(~{result.prompt_chars} chars prompt)"
        )
        if result.prompt_tokens is not None:
            print(f"Actual prompt tokens: {result.prompt_tokens}")
        if result.completion_tokens is not None:
            print(f"Completion tokens: {result.completion_tokens}")
            if result.completion_tokens < int(result.max_output_tokens * 0.7):
                print(
                    f"Note: completion ended early (requested {result.max_output_tokens}, got {result.completion_tokens})"
                )
        if result.total_tokens is not None:
            print(f"Total tokens: {result.total_tokens}")
        print(f"Duration: {result.duration_s:.2f}s")
        if result.tokens_per_s is not None:
            print(f"Tokens/s (total): {result.tokens_per_s:.2f}")
        if result.completion_tokens_per_s is not None:
            print(f"Tokens/s (completion): {result.completion_tokens_per_s:.2f}")
        if result.avg_gpu_util is not None:
            print(f"Avg GPU util: {result.avg_gpu_util:.1f}%")
        if result.avg_gpu_power_w is not None:
            print(f"Avg GPU power: {result.avg_gpu_power_w:.1f}W")
        if result.avg_gpu_power_percent_of_limit is not None:
            print(f"Avg GPU power cap: {result.avg_gpu_power_percent_of_limit:.1f}%")
        if result.gpu_energy_wh is not None:
            print(f"GPU energy: {result.gpu_energy_wh:.4f}Wh")
        if result.tokens_per_s_per_w is not None:
            print(f"Tokens/s/W: {result.tokens_per_s_per_w:.4f}")
        if result.avg_cpu_percent is not None:
            print(f"Avg CPU: {result.avg_cpu_percent:.1f}%")
        if result.avg_mem_percent is not None:
            print(f"Avg Mem: {result.avg_mem_percent:.1f}%")
        if result.avg_process_cpu_percent is not None:
            print(f"Avg Proc CPU: {result.avg_process_cpu_percent:.1f}%")
        if result.avg_process_rss_mb is not None:
            print(f"Avg Proc RSS: {result.avg_process_rss_mb:.0f}MB")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(benchmark.results_to_json(results))
        print(f"Wrote {args.json_out}")

    if results:
        print("-")
        print("Table:")
        print(format_results_table(results))

    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    prompt_sizes = benchmark.parse_size_list(args.prompt_sizes)
    tasks = [t.strip() for t in str(getattr(args, "task", "coding")).split(",") if t.strip()]
    if not tasks:
        tasks = ["coding"]
    output_tokens_list = parse_int_list_csv(getattr(args, "output_tokens", None))
    if not output_tokens_list:
        output_tokens_list = [int(args.max_output_tokens)]
    extra: Optional[Dict[str, Any]] = None
    if args.extra:
        extra = json.loads(args.extra)

    # Prefer reusing an existing server.
    endpoint = f"http://{args.host}:{args.port}"
    existing_pid = find_llama_server_on_port(args.port)

    started_proc: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    try:
        if existing_pid:
            pid = existing_pid
            print(f"Reusing existing llama-server on port {args.port} (pid {pid})")
        else:
            # If something is already listening on the requested port, try to reuse it
            # as an OpenAI-compatible endpoint (e.g. llama-run proxy).
            ok, _ = utils.check_port_available(args.port)
            if not ok and wait_for_server_ready(endpoint, timeout_s=2.0):
                pid = utils.find_pid_by_port(args.port)
                print(f"Reusing existing server endpoint on port {args.port} (pid {pid or 'unknown'})")

            running = utils.find_llama_server_processes()
            if running and not args.allow_multi_server:
                # If exactly one server is running, try to reuse it.
                if len(running) == 1:
                    only_pid, only_cmd = running[0]
                    inferred = infer_port_from_cmdline(only_cmd or "")
                    if inferred:
                        endpoint = f"http://{args.host}:{inferred}"
                        print(f"Reusing existing llama-server pid {only_pid} (port {inferred})")
                        pid = only_pid
                    else:
                        raise RuntimeError(
                            "Detected a running llama-server but could not infer its port. "
                            "Specify --port, or pass --allow-multi-server to start a new one."
                        )
                else:
                    raise RuntimeError(
                        "Detected running llama-server processes. Specify --port to reuse one, "
                        "or pass --allow-multi-server to start another."
                    )

        if pid is None:
            # Start a new server.
            ok, reason = utils.check_port_available(args.port)
            if not ok:
                raise RuntimeError(reason or "port unavailable")

            if not args.skip_vram_check:
                ok, reason = utils.check_vram_available(float(args.min_vram_free_mb))
                if not ok:
                    raise RuntimeError(reason)

            llama_cpp_dir = args.llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
            llama_server = args.llama_server or utils.find_llama_server(llama_cpp_dir)
            if not llama_server:
                raise RuntimeError(
                    "llama-server not found. Provide --llama-server or --llama-cpp-dir, "
                    "or ensure llama-server is on PATH."
                )

            model_path = utils.find_model_path(
                args.model,
                args.model_dir,
                preferred_files=utils.DEFAULT_MODEL_FILES,
                preferred_substrings=utils.DEFAULT_MODEL_SUBSTRINGS,
            )
            if not model_path:
                raise RuntimeError("Model not found. Provide --model or place a .gguf in ~/models.")

            model_alias = args.model_alias or utils.default_model_alias(model_path)
            cmd = [
                llama_server,
                "--model",
                model_path,
                "--alias",
                model_alias or utils.default_model_alias(model_path) or "model",
                "--port",
                str(args.port),
                "--host",
                args.host,
                "--fit",
                args.fit,
                "--temp",
                str(args.temp),
                "--top-p",
                str(args.top_p),
                "--min-p",
                str(args.min_p),
                "--ctx-size",
                str(args.ctx_size),
                "--batch-size",
                str(args.batch_size),
                "--ubatch-size",
                str(args.ubatch_size),
                "--cache-type-k",
                args.cache_type_k,
                "--cache-type-v",
                args.cache_type_v,
                "--flash-attn",
                args.flash_attn,
            ]
            if args.kv_unified == "on":
                cmd.append("--kv-unified")
            if args.jinja == "on":
                cmd.append("--jinja")
                if args.chat_template_kwargs:
                    cmd.extend(["--chat-template-kwargs", args.chat_template_kwargs])
                elif args.enable_thinking:
                    cmd.extend(["--chat-template-kwargs", f'{{"enable_thinking": {args.enable_thinking}}}'])

            extra_args = list(args.extra_args or [])
            if extra_args and extra_args[0] == "--":
                extra_args = extra_args[1:]
            cmd.extend(extra_args)

            print("Starting llama-server...")
            started_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid = started_proc.pid

            if not utils.wait_for_port(args.host, args.port, timeout_s=float(args.startup_timeout)):
                raise RuntimeError("server did not start listening in time")
            if not wait_for_server_ready(endpoint, timeout_s=float(args.startup_timeout)):
                print("Warning: server did not report ready endpoints before timeout; continuing...", file=sys.stderr)

        metrics_url = args.metrics_url or (endpoint.rstrip("/") + "/metrics")

        results = []
        for task in tasks:
            for size in prompt_sizes:
                for out_tokens in output_tokens_list:
                    print(f"Starting benchmark: task={task} in={size} out={out_tokens}")

                    t0 = time.time()
                    progress_state = {"last": 0.0}

                    def on_sample(sample: utils.Sample) -> None:
                        if args.progress != "on":
                            return
                        now = time.time()
                        if (now - float(progress_state["last"])) < float(args.progress_interval):
                            return
                        progress_state["last"] = now
                        elapsed = now - t0
                        sys.stdout.write(
                            f"[progress] task={task} in={size} out={out_tokens} elapsed={elapsed:.1f}s {utils.format_sample_line(sample)}\n"
                        )
                        sys.stdout.flush()

                    deadline = time.time() + float(getattr(args, "retry_503", 0.0) or 0.0)
                    attempt = 0
                    while True:
                        attempt += 1
                        sampler = utils.MetricsSampler(
                            interval=args.interval,
                            pid=pid,
                            metrics_url=metrics_url,
                        )
                        sampler.on_sample = on_sample  # type: ignore[attr-defined]
                        try:
                            result = benchmark.run_single_benchmark(
                                endpoint=endpoint,
                                task=task,
                                prompt_tokens_target=size,
                                max_output_tokens=out_tokens,
                                temperature=args.temperature,
                                mode=args.mode,
                                model=args.api_model,
                                extra=extra,
                                sampler=sampler,
                            )
                            break
                        except Exception as exc:
                            code = getattr(getattr(exc, "response", None), "status_code", None)
                            if code == 503 and time.time() < deadline:
                                print(f"Server returned 503 (attempt {attempt}); retrying...", file=sys.stderr)
                                time.sleep(1.0)
                                continue
                            raise
                    results.append(result)

        for result in results:
            print("-")
            print(
                f"Scenario: task={result.task} in={result.prompt_target_tokens} out={result.max_output_tokens} "
                f"(~{result.prompt_chars} chars prompt)"
            )
            if result.prompt_tokens is not None:
                print(f"Actual prompt tokens: {result.prompt_tokens}")
            if result.completion_tokens is not None:
                print(f"Completion tokens: {result.completion_tokens}")
                if result.completion_tokens < int(result.max_output_tokens * 0.7):
                    print(
                        f"Note: completion ended early (requested {result.max_output_tokens}, got {result.completion_tokens})"
                    )
            if result.total_tokens is not None:
                print(f"Total tokens: {result.total_tokens}")
            print(f"Duration: {result.duration_s:.2f}s")
            if result.tokens_per_s is not None:
                print(f"Tokens/s (total): {result.tokens_per_s:.2f}")
            if result.completion_tokens_per_s is not None:
                print(f"Tokens/s (completion): {result.completion_tokens_per_s:.2f}")
            if result.avg_gpu_util is not None:
                print(f"Avg GPU util: {result.avg_gpu_util:.1f}%")
            if result.avg_gpu_power_w is not None:
                print(f"Avg GPU power: {result.avg_gpu_power_w:.1f}W")
            if result.avg_gpu_power_percent_of_limit is not None:
                print(f"Avg GPU power cap: {result.avg_gpu_power_percent_of_limit:.1f}%")
            if result.gpu_energy_wh is not None:
                print(f"GPU energy: {result.gpu_energy_wh:.4f}Wh")
            if result.tokens_per_s_per_w is not None:
                print(f"Tokens/s/W: {result.tokens_per_s_per_w:.4f}")
            if result.avg_cpu_percent is not None:
                print(f"Avg CPU: {result.avg_cpu_percent:.1f}%")
            if result.avg_mem_percent is not None:
                print(f"Avg Mem: {result.avg_mem_percent:.1f}%")
            if result.avg_process_cpu_percent is not None:
                print(f"Avg Proc CPU: {result.avg_process_cpu_percent:.1f}%")
            if result.avg_process_rss_mb is not None:
                print(f"Avg Proc RSS: {result.avg_process_rss_mb:.0f}MB")

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(benchmark.results_to_json(results))
            print(f"Wrote {args.json_out}")

        if results:
            print("-")
            print("Table:")
            print(format_results_table(results))
        return 0
    finally:
        if started_proc and not args.keep_server:
            try:
                started_proc.send_signal(signal.SIGINT)
                started_proc.wait(timeout=15)
            except Exception:
                try:
                    started_proc.kill()
                except Exception:
                    pass


def cmd_autotune(args: argparse.Namespace) -> int:
    batch_sizes = autotune.parse_int_list(args.batch_sizes)
    ubatch_sizes = autotune.parse_int_list(args.ubatch_sizes)
    extra_request: Optional[Dict[str, Any]] = None
    if args.extra_request:
        extra_request = json.loads(args.extra_request)

    chat_kwargs = args.chat_template_kwargs
    if not chat_kwargs and args.enable_thinking and args.jinja == "on":
        chat_kwargs = f'{{\"enable_thinking\": {args.enable_thinking}}}'

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    try:
        best, results = autotune.run_autotune(
            llama_cpp_dir=args.llama_cpp_dir,
            llama_server=args.llama_server,
            model=args.model,
            model_dir=args.model_dir,
            host=args.host,
            ctx_size=args.ctx_size,
            batch_sizes=batch_sizes,
            ubatch_sizes=ubatch_sizes,
            duration_s=args.duration,
            prompt_tokens=args.prompt_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            mode="chat",
            model_name=None,
            objective=args.objective,
            interval=args.interval,
            startup_timeout=args.startup_timeout,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
            fit=args.fit,
            flash_attn=args.flash_attn,
            kv_unified=args.kv_unified,
            jinja=args.jinja,
            chat_template_kwargs=chat_kwargs,
            extra_request=extra_request,
            extra_args=extra_args,
            output_path=args.output,
            min_vram_free_mb=float(args.min_vram_free_mb),
            skip_vram_check=args.skip_vram_check,
            allow_multi_server=args.allow_multi_server,
        )
    except Exception as exc:
        print(f"Autotune failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No successful results.")
        return 1

    if best:
        print("Best settings:")
        print(f"  score ({best.objective}): {best.score:.4f}")
        if best.tokens_per_s is not None:
            print(f"  tokens/s: {best.tokens_per_s:.2f}")
        if best.tokens_per_s_per_w is not None:
            print(f"  tokens/s/W: {best.tokens_per_s_per_w:.4f}")
        print(f"  batch: {best.config.get('batch_size')} ubatch: {best.config.get('ubatch_size')}")
        print(f"Saved best settings to {args.output}")

    return 0


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "server":
        return cmd_server(args)
    if args.command == "monitor":
        return cmd_monitor(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)
    if args.command == "autotune":
        return cmd_autotune(args)
    if args.command == "bench":
        return cmd_bench(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
