import argparse
from collections import deque
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from tools.llama_bench import utils
from tools.llama_run.tui import TuiState, run_tui
from tools.llama_run.proxy import LlamaMeterProxy
from tools.llama_run.logperf import LogPerfMeter

# Minimum context size - model will never be loaded with less than this
MIN_CONTEXT_SIZE = 4096


def _wait_for_server(host: str, port: int, timeout_s: float = 60.0) -> bool:
    """Wait for server to be ready by polling the port."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _wait_for_models_ready(host: str, port: int, timeout_s: float = 180.0) -> bool:
    """Wait until llama-server reports a loaded model via /v1/models."""
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-run",
        description="Run llama.cpp llama-server with live logs and rolling stats.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run llama-server and print rolling stats")
    run.add_argument("--llama-server", help="Path to llama-server binary (otherwise autodetect)")
    run.add_argument("--llama-cpp-dir", help="llama.cpp directory (LLAMA_CPP_DIR env var)")
    run.add_argument("--model", help="Path to .gguf model (MODEL env var)")
    run.add_argument("--model-dir", help="Search for .gguf models in this directory")
    run.add_argument("--model-alias", help="Model alias (defaults to model filename)")
    run.add_argument("--host", default="127.0.0.1", help="Bind host")
    run.add_argument("--port", type=int, default=11433, help="Server port")

    # common knobs (kept in sync with tools/llama_bench)
    run.add_argument(
        "--ctx-size",
        type=int,
        default=None,
        help="Maximum context size cap. If omitted, default is the model's GGUF context_length.",
    )
    run.add_argument(
        "--max-model-ctx",
        type=int,
        default=None,
        help="Maximum context size the model supports. If omitted, try to read it from GGUF metadata; otherwise fall back to 128000.",
    )
    run.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    run.add_argument("--ubatch-size", type=int, default=1024, help="Micro-batch size")
    run.add_argument("--cache-type-k", default="q8_0", help="KV cache type for K")
    run.add_argument("--cache-type-v", default="q8_0", help="KV cache type for V")
    run.add_argument("--fit", choices=["on", "off"], default="on", help="Enable context fitting")
    run.add_argument("--flash-attn", choices=["on", "off"], default="on", help="Enable flash attention")
    run.add_argument("--kv-unified", choices=["on", "off"], default="on", help="Enable unified KV")
    run.add_argument("--jinja", choices=["on", "off"], default="on", help="Enable Jinja chat templates")
    run.add_argument("--chat-template-kwargs", help="JSON for --chat-template-kwargs")
    run.add_argument(
        "--enable-thinking",
        choices=["true", "false"],
        default="true",
        help="Enable thinking flag (if using Jinja templates)",
    )
    run.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    run.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    run.add_argument("--min-p", type=float, default=0.01, help="Min-p sampling")

    # live stats
    run.add_argument("--interval", type=float, default=1.0, help="Metrics sampling interval (seconds)")
    run.add_argument(
        "--ui",
        choices=["tui", "plain"],
        default="tui",
        help="UI mode (tui is htop-like, plain is log+stats lines)",
    )
    run.add_argument(
        "--proxy",
        choices=["on", "off"],
        default="on",
        help="Run a local proxy on --port to meter tokens (recommended)",
    )
    run.add_argument("--stats-interval", type=float, default=10.0, help="How often to print stats in plain mode")
    run.add_argument("--metrics-url", help="Prometheus metrics endpoint (default: http://HOST:PORT/metrics)")
    run.add_argument("--no-gpu", action="store_true", help="Disable GPU sampling")
    run.add_argument("--test-timeout", type=float, default=0, help="Auto-stop after N seconds (for testing)")
    run.add_argument(
        "--auto-optimize-context",
        action="store_true",
        default=True,
        help="After model loads, increase context (up to requested/model max) if there's VRAM headroom (default: on)",
    )
    run.add_argument(
        "--no-auto-optimize-context",
        action="store_false",
        dest="auto_optimize_context",
        help="Disable post-load context optimization",
    )
    run.add_argument(
        "--optimize-context-threshold-mb",
        type=float,
        default=2048.0,
        help="Minimum 'wasted' VRAM (above reserve) before optimizing context size (default: 2048 MB)",
    )

    run.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to llama-server (use after --)",
    )

    stop_all = subparsers.add_parser("stop-all", help="Stop all running llama-server processes")
    stop_all.add_argument("--dry-run", action="store_true", help="Only print what would be stopped")
    stop_all.add_argument(
        "--force",
        action="store_true",
        help="Escalate to SIGKILL if needed",
    )
    stop_all.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGINT/SIGTERM",
    )

    return parser


def _resolve_metrics_url(args: argparse.Namespace) -> str:
    if args.metrics_url:
        return args.metrics_url
    return f"http://{args.host}:{args.port}/metrics"


def _format_window(label: str, summary: utils.SampleSummary) -> str:
    # NOTE: token rates/counts come from llama.cpp logs, not /metrics.
    cpu = utils.format_percent(summary.avg_cpu_percent)
    mem = utils.format_percent(summary.avg_mem_percent)
    rss = utils.format_number(summary.avg_process_rss_mb, "MB", precision=0)
    gpu = utils.format_percent(summary.avg_gpu_util)
    pwr = utils.format_number(summary.avg_gpu_power_w, "W", precision=1)
    cap = utils.format_percent(summary.avg_gpu_power_percent_of_limit)
    vram = "n/a"
    if summary.avg_gpu_mem_used_mb is not None and summary.avg_gpu_mem_total_mb is not None:
        vram = f"{summary.avg_gpu_mem_used_mb/1024.0:.1f}/{summary.avg_gpu_mem_total_mb/1024.0:.1f}GB"

    cpu_w = utils.format_number(summary.cpu_power_w, "W", precision=1)
    pc_w = utils.format_number(summary.pc_power_w, "W", precision=1)

    def fmt_wh(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        wh = float(value)
        if wh >= 1000.0:
            return f"{wh/1000.0:.3f}kWh"
        return f"{wh:.2f}Wh"

    cpu_wh = fmt_wh(summary.cpu_energy_wh)
    gpu_wh = fmt_wh(summary.gpu_energy_wh)
    pc_wh = fmt_wh(summary.pc_energy_wh)
    pc_tag = "pc" if summary.pc_energy_source != "estimated" else "pc*"
    return (
        f"{label}: cpu {cpu} mem {mem} rss {rss} | "
        f"gpu {gpu} cap {cap} pwr {pwr} vram {vram} | cpu_pwr {cpu_w} {pc_tag}_pwr {pc_w} | "
        f"energy cpu {cpu_wh} gpu {gpu_wh} {pc_tag} {pc_wh}"
    )


def get_system_info() -> str:
    """Get CPU and GPU information for display."""
    cpu_name, cpu_count = utils.get_cpu_info()
    
    gpus = utils.get_gpu_metrics()
    gpu_info = []
    if gpus:
        for gpu in gpus:
            if gpu.mem_total_mb:
                gpu_str = f"{gpu.name} ({gpu.mem_total_mb/1024.0:.1f}GB)"
            else:
                gpu_str = gpu.name
            gpu_info.append(gpu_str)
    
    if gpu_info:
        gpu_str = "| ".join(gpu_info)
    else:
        gpu_str = "n/a"

    # Get system memory info
    try:
        import subprocess
        result = subprocess.run(
            ["free", "-g"],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Mem:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_total = parts[1]
                        mem_str = f"{mem_total}GB"
                    else:
                        mem_str = "n/a"
                    break
            else:
                mem_str = "n/a"
        else:
            mem_str = "n/a"
    except Exception:
        mem_str = "n/a"

    return f"CPU: {cpu_name} ({cpu_count} cores) | RAM: {mem_str} | GPU: {gpu_str}"


def _build_server_cmd(
    llama_server: str,
    model_path: str,
    backend_port: int,
    ctx_size: int,
    args: argparse.Namespace,
    model_alias: Optional[str] = None,
) -> list:
    """Build the llama-server command with given context size."""
    cmd = [
        llama_server,
        "--model",
        model_path,
        "--port",
        str(backend_port),
        "--fit",
        args.fit,
        "--temp",
        str(args.temp),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--ctx-size",
        str(ctx_size),
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
        "--host",
        "127.0.0.1" if args.proxy == "on" else args.host,
    ]
    if model_alias:
        cmd.extend(["--alias", model_alias])
    if args.kv_unified == "on":
        cmd.append("--kv-unified")
    if args.jinja == "on":
        cmd.append("--jinja")
        if args.chat_template_kwargs:
            cmd.extend(["--chat-template-kwargs", args.chat_template_kwargs])
        else:
            thinking = "true" if args.enable_thinking == "true" else "false"
            cmd.extend(["--chat-template-kwargs", f'{{"enable_thinking": {thinking}}}'])

    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    return cmd


def cmd_run(args: argparse.Namespace) -> int:
    llama_cpp_dir = args.llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    llama_server = args.llama_server or utils.find_llama_server(llama_cpp_dir)
    if not llama_server:
        print(
            "llama-server not found. Provide --llama-server or --llama-cpp-dir, or ensure llama-server is on PATH.",
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

    # Determine model context limit (never set ctx-size above this).
    detected_model_ctx = utils.get_gguf_context_length(model_path)
    if detected_model_ctx:
        if args.max_model_ctx is None:
            args.max_model_ctx = int(detected_model_ctx)
            print(f"Info: Model GGUF metadata reports context_length={args.max_model_ctx:,}")
        else:
            if int(args.max_model_ctx) > int(detected_model_ctx):
                print(
                    f"Warning: --max-model-ctx ({args.max_model_ctx:,}) exceeds GGUF context_length "
                    f"({int(detected_model_ctx):,}). Capping to {int(detected_model_ctx):,}.",
                    file=sys.stderr,
                )
                args.max_model_ctx = int(detected_model_ctx)
    else:
        if args.max_model_ctx is None:
            args.max_model_ctx = 128000
            print(
                "Warning: Could not read model context_length from GGUF metadata; "
                "using conservative default --max-model-ctx 128000. Override with --max-model-ctx if needed.",
                file=sys.stderr,
            )

    proxy: Optional[LlamaMeterProxy] = None
    backend_port = args.port
    if args.proxy == "on":
        # Reserve the public port for our proxy and run the actual llama-server on a free port.
        ok, reason = utils.check_port_available(args.port)
        if not ok:
            print(reason, file=sys.stderr)
            return 1
        backend_port = utils.find_free_port()
    else:
        ok, reason = utils.check_port_available(args.port)
        if not ok:
            print(reason, file=sys.stderr)
            return 1

    model_alias = args.model_alias or utils.default_model_alias(model_path)
    
    # Determine the desired context cap:
    # - default: model max context_length
    # - if user specifies --ctx-size: treat it as a cap
    user_ctx_cap = args.ctx_size if args.ctx_size is not None else int(args.max_model_ctx)
    desired_ctx_cap = int(min(int(user_ctx_cap), int(args.max_model_ctx)))
    if desired_ctx_cap < MIN_CONTEXT_SIZE:
        print(
            f"Notice: Requested/capped ctx-size ({desired_ctx_cap:,}) is below minimum ({MIN_CONTEXT_SIZE:,}). "
            f"Using {MIN_CONTEXT_SIZE:,} instead.",
            file=sys.stderr,
        )
        desired_ctx_cap = MIN_CONTEXT_SIZE

    # Calculate a conservative pre-load context size based on available VRAM.
    # This is used as a safe starting point; we may increase it after the model is loaded.
    max_context, ctx_msg = utils.calculate_max_context_size(
        model_path=model_path,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        reserve_vram_mb=utils.DEFAULT_VRAM_RESERVE_MB,
    )
    if max_context is None:
        print(f"Warning: Could not estimate VRAM-based max context. {ctx_msg}", file=sys.stderr)
        effective_ctx_size = desired_ctx_cap
    else:
        # Start at the conservative estimate, but never exceed desired cap.
        if int(max_context) < int(desired_ctx_cap):
            print(
                f"Warning: Desired ctx-size ({desired_ctx_cap:,}) exceeds conservative VRAM estimate ({int(max_context):,}). "
                f"Starting at {int(max_context):,} and will try to raise after model load. {ctx_msg}",
                file=sys.stderr,
            )
        else:
            print(f"Info: {ctx_msg}")
        effective_ctx_size = int(min(int(max_context), int(desired_ctx_cap)))
    
    # Ensure minimum context size (again, after VRAM estimation)
    if effective_ctx_size < MIN_CONTEXT_SIZE:
        print(
            f"Notice: Context size ({effective_ctx_size:,}) is below minimum ({MIN_CONTEXT_SIZE:,}). "
            f"Using {MIN_CONTEXT_SIZE:,} instead.",
            file=sys.stderr,
        )
        effective_ctx_size = MIN_CONTEXT_SIZE

    # Enforce maximum model context size - NEVER exceed what the model supports
    if effective_ctx_size > int(args.max_model_ctx):
        print(
            f"Warning: Selected context ({effective_ctx_size:,}) exceeds model's maximum "
            f"({int(args.max_model_ctx):,}). Using {int(args.max_model_ctx):,} instead.",
            file=sys.stderr,
        )
        effective_ctx_size = int(args.max_model_ctx)
    
    # Build command and start server
    cmd = _build_server_cmd(
        llama_server=llama_server,
        model_path=model_path,
        backend_port=backend_port,
        ctx_size=effective_ctx_size,
        args=args,
        model_alias=model_alias,
    )

    print(get_system_info())
    print("Launching llama-server:")
    print(" ".join(cmd))
    print("---")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for model load and check if we can optimize context.
    # IMPORTANT: The TCP port can open before the model is loaded; /v1/models is the readiness signal.
    server_host = "127.0.0.1" if args.proxy == "on" else args.host
    desired_ctx = int(desired_ctx_cap)

    if _wait_for_models_ready(server_host, backend_port, timeout_s=180.0):
        free_mb, free_msg = utils.get_max_free_vram_mb()
        if free_mb is None:
            # Can't make a VRAM-based decision, but we can still keep the initial ctx.
            pass
        else:
            # If we started below the desired cap and have headroom, try to increase context.
            # Primary strategy: try the desired cap directly (model max by default). If it fails
            # to become ready, fall back to the previous working context.
            headroom_mb = float(free_mb) - float(utils.DEFAULT_VRAM_RESERVE_MB)
            # If we started below the desired context cap, try the cap directly.
            # This is the behavior you want for "use model max ctx unless it OOMs".
            if args.auto_optimize_context and effective_ctx_size < desired_ctx:
                candidate_ctx = int(desired_ctx)
                if candidate_ctx > int(args.max_model_ctx):
                    candidate_ctx = int(args.max_model_ctx)

                print(
                    f"\n[context-optimize] post-load VRAM: {free_mb:.0f}MB free (headroom {headroom_mb:.0f}MB). "
                    f"Trying ctx-size {effective_ctx_size:,} -> {candidate_ctx:,} (cap {desired_ctx:,}, model max {int(args.max_model_ctx):,}). {free_msg}"
                )
                print(f"[context-optimize] Restarting server with higher context size: {candidate_ctx:,}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                
                # Rebuild command with optimized context
                cmd = _build_server_cmd(
                    llama_server=llama_server,
                    model_path=model_path,
                    backend_port=backend_port,
                    ctx_size=candidate_ctx,
                    args=args,
                    model_alias=model_alias,
                )
                
                print("\nRelaunching llama-server with optimized context:")
                print(" ".join(cmd))
                print("---")
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                effective_ctx_size = candidate_ctx

                # Wait again for the restarted server to finish loading.
                if not _wait_for_models_ready(server_host, backend_port, timeout_s=180.0):
                    print(
                        f"Warning: Server did not become ready after trying ctx-size {candidate_ctx:,}; "
                        f"falling back to {int(effective_ctx_size):,}.",
                        file=sys.stderr,
                    )
                    try:
                        proc.terminate()
                        proc.wait(timeout=10)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass

                    # Fall back to the previous (conservative) context.
                    fallback_ctx = int(min(int(max_context or effective_ctx_size), int(desired_ctx)))
                    if fallback_ctx < MIN_CONTEXT_SIZE:
                        fallback_ctx = MIN_CONTEXT_SIZE
                    cmd = _build_server_cmd(
                        llama_server=llama_server,
                        model_path=model_path,
                        backend_port=backend_port,
                        ctx_size=fallback_ctx,
                        args=args,
                        model_alias=model_alias,
                    )
                    print("\nRelaunching llama-server with fallback context:")
                    print(" ".join(cmd))
                    print("---")
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    effective_ctx_size = fallback_ctx
                    if not _wait_for_models_ready(server_host, backend_port, timeout_s=180.0):
                        print("Error: Server failed to become ready after fallback", file=sys.stderr)
                        return 1
    else:
        print("Warning: Server did not become ready (/v1/models) within timeout; skipping context optimization", file=sys.stderr)

    # Always print a single final context decision line.
    cons_str = "n/a" if max_context is None else f"{int(max_context):,}"
    sys.stdout.write(
        f"[ctx] selected: {effective_ctx_size:,} | cap: {int(desired_ctx_cap):,} | "
        f"model_max: {int(args.max_model_ctx):,} | conservative_est: {cons_str}\n"
    )
    sys.stdout.flush()

    logs = deque(maxlen=500)
    perf = LogPerfMeter(retention_s=16 * 60.0)
    stop_logs = threading.Event()

    def _pump_logs() -> None:
        if not proc.stdout:
            return
        try:
            for line in proc.stdout:
                if stop_logs.is_set():
                    break
                # Keep a tail buffer for TUI; only print in plain mode.
                logs.append(line.rstrip("\n"))
                perf.add_log_line(line)
                if args.ui == "plain":
                    sys.stdout.write(line)
                    sys.stdout.flush()
        except Exception:
            return

    log_thread = threading.Thread(target=_pump_logs, daemon=True)
    log_thread.start()

    if args.proxy == "on":
        proxy = LlamaMeterProxy(
            host=args.host,
            port=args.port,
            backend_host="127.0.0.1",
            backend_port=int(backend_port),
        )
        proxy.start()
        print(f"Proxy listening on http://{args.host}:{args.port} -> backend http://127.0.0.1:{backend_port}")

    # In proxy mode, /metrics is served by the proxy and always includes token counters.
    if not args.metrics_url and args.proxy == "on":
        args.metrics_url = f"http://{args.host}:{args.port}/metrics"
    metrics_url = _resolve_metrics_url(args)
    sampler = utils.MetricsSampler(
        interval=args.interval,
        pid=proc.pid,
        metrics_url=metrics_url,
        include_gpu=(not args.no_gpu),
    )
    sampler.start()

    started_at = time.time()

    want_tui = args.ui == "tui" and sys.stdin.isatty() and sys.stdout.isatty()
    if args.ui == "tui" and not want_tui:
        print("note: no TTY detected; falling back to --ui plain", file=sys.stderr)
        want_tui = False

    # Setup timeout if specified
    test_deadline = None
    if args.test_timeout > 0:
        test_deadline = time.time() + args.test_timeout
        print(f"[test] Auto-stop in {args.test_timeout}s")

    def _stop() -> None:
        try:
            if proxy:
                proxy.stop()
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    # Start timeout timer if specified
    timeout_timer = None
    if test_deadline:
        def _timeout_handler():
            time.sleep(args.test_timeout)
            print("\n[test] Timeout reached, stopping server...")
            _stop()
        timeout_timer = threading.Thread(target=_timeout_handler, daemon=True)
        timeout_timer.start()

    try:
        if want_tui:
            endpoint = f"http://{args.host}:{args.port}"

            def _alive() -> bool:
                return proc.poll() is None

            title = "llama-run"
            state = TuiState(
                title=title,
                endpoint=endpoint,
                pid=int(proc.pid),
                started_at=started_at,
                sampler=sampler,
                perf=perf,
                logs=logs,
                max_context_size=effective_ctx_size,
                last_ctx_tokens=0,
                system_info=get_system_info(),
            )
            run_tui(state, stop_cb=_stop, alive_cb=_alive, proxy=proxy)
        else:
            last_print = 0.0
            last_ctx_tokens = 0
            while proc.poll() is None:
                time.sleep(0.25)
                now = time.time()
                if args.stats_interval and (now - last_print) >= float(args.stats_interval):
                    last_print = now
                    all_sum = sampler.summarize_window(None)
                    now_sum = sampler.summarize_window(5.0)
                    m1 = sampler.summarize_window(60.0)
                    m15 = sampler.summarize_window(15.0 * 60.0)

                    p_all, g_all = perf.totals()
                    now_p = perf.last()
                    m1_p = perf.window(60.0)
                    m15_p = perf.window(15.0 * 60.0)
                    all_p = perf.window(None)

                    # Context usage (tokens used by the most recent request).
                    # Prefer proxy metering; fall back to llama-server log parsing.
                    last_ctx_tokens = 0
                    if proxy is not None:
                        try:
                            last_ctx_tokens = int(proxy.meter.get_current_context())
                        except Exception:
                            last_ctx_tokens = 0
                    if last_ctx_tokens <= 0:
                        last_ctx_tokens = int(now_p.total_tokens)

                    def fmt_tps(v):
                        return "n/a" if v is None else f"{v:.1f}"

                    def tok_line(label: str, w):
                        return (
                            f"[tok] {label}: p_tok/s {fmt_tps(w.prompt_tps)} g_tok/s {fmt_tps(w.gen_tps)} "
                            f"p_tok {w.prompt_tokens} g_tok {w.gen_tokens}"
                        )

                    # Calculate context usage percentage
                    if effective_ctx_size > 0:
                        ctx_pct = (last_ctx_tokens / effective_ctx_size) * 100
                        ctx_info = f"{last_ctx_tokens:,} / {effective_ctx_size:,} ({ctx_pct:.1f}%)"
                    else:
                        ctx_info = f"{last_ctx_tokens:,} / n/a"

                    sys.stdout.write(
                        f"[tok] totals: prompt {p_all} gen {g_all} total {p_all + g_all}\n"
                        + f"[ctx] max: {effective_ctx_size:,} | last request: {ctx_info}\n"
                        + tok_line("now", now_p)
                        + "\n"
                        + tok_line("1m", m1_p)
                        + "\n"
                        + tok_line("15m", m15_p)
                        + "\n"
                        + tok_line("all", all_p)
                        + "\n"
                        + get_system_info()
                        + "\n"
                        + "[sys] "
                        + _format_window("now", now_sum)
                        + "\n"
                        + "[sys] "
                        + _format_window("1m", m1)
                        + "\n"
                        + "[sys] "
                        + _format_window("15m", m15)
                        + "\n"
                        + "[sys] "
                        + _format_window("all", all_sum)
                        + "\n"
                    )
                    sys.stdout.flush()
                    sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\nStopping server...\n")
        sys.stdout.flush()
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        sampler.stop()
        stop_logs.set()
        if proxy:
            try:
                proxy.stop()
            except Exception:
                pass
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        log_thread.join(timeout=2.0)

    rc = proc.wait()
    sys.stdout.write("---\n")
    sys.stdout.write(f"llama-server exited with code {rc}\n")
    sys.stdout.write(get_system_info())
    sys.stdout.write("\n")
    sys.stdout.write("Stats summary:\n")
    
    # Context usage (tokens used by the most recent request).
    # Prefer proxy metering; fall back to the last log-perf sample.
    final_last_ctx = 0
    if proxy is not None:
        try:
            final_last_ctx = int(proxy.meter.get_current_context())
        except Exception:
            final_last_ctx = 0
    if final_last_ctx <= 0:
        try:
            final_last_ctx = int(perf.last().total_tokens)
        except Exception:
            final_last_ctx = 0

    # Calculate final context usage
    if effective_ctx_size > 0:
        final_ctx_pct = (final_last_ctx / effective_ctx_size) * 100
        final_ctx_info = f"{final_last_ctx:,} / {effective_ctx_size:,} ({final_ctx_pct:.1f}%)"
    else:
        final_ctx_info = f"{final_last_ctx:,} / n/a"

    sys.stdout.write(f"[ctx] max context: {effective_ctx_size:,} | last request usage: {final_ctx_info}\n")
    
    now_p = perf.last()
    m1_p = perf.window(60.0)
    m15_p = perf.window(15.0 * 60.0)
    all_p = perf.window(None)

    def fmt_tps(v):
        return "n/a" if v is None else f"{v:.1f}"

    sys.stdout.write(
        f"[tok] now: p_tok/s {fmt_tps(now_p.prompt_tps)} g_tok/s {fmt_tps(now_p.gen_tps)} p_tok {now_p.prompt_tokens} g_tok {now_p.gen_tokens}\n"
    )
    sys.stdout.write(
        f"[tok] 1m:  p_tok/s {fmt_tps(m1_p.prompt_tps)} g_tok/s {fmt_tps(m1_p.gen_tps)} p_tok {m1_p.prompt_tokens} g_tok {m1_p.gen_tokens}\n"
    )
    sys.stdout.write(
        f"[tok] 15m: p_tok/s {fmt_tps(m15_p.prompt_tps)} g_tok/s {fmt_tps(m15_p.gen_tps)} p_tok {m15_p.prompt_tokens} g_tok {m15_p.gen_tokens}\n"
    )
    sys.stdout.write(
        f"[tok] all: p_tok/s {fmt_tps(all_p.prompt_tps)} g_tok/s {fmt_tps(all_p.gen_tps)} p_tok {all_p.prompt_tokens} g_tok {all_p.gen_tokens}\n"
    )
    sys.stdout.write("[sys] " + _format_window("now", sampler.summarize_window(5.0)) + "\n")
    sys.stdout.write("[sys] " + _format_window("1m", sampler.summarize_window(60.0)) + "\n")
    sys.stdout.write("[sys] " + _format_window("15m", sampler.summarize_window(15.0 * 60.0)) + "\n")
    sys.stdout.write("[sys] " + _format_window("all", sampler.summarize_window(None)) + "\n")
    sys.stdout.flush()
    return rc


def cmd_stop_all(args: argparse.Namespace) -> int:
    procs = utils.find_llama_server_processes()
    # Filter out our own process (if the user runs llama-run from a cmd containing llama-server).
    self_pid = os.getpid()
    targets = [(pid, cmd) for pid, cmd in procs if pid and pid != self_pid]

    if not targets:
        print("No running llama-server processes found")
        return 0

    print("Found llama-server processes:")
    for pid, cmd in targets:
        line = cmd or "llama-server"
        print(f"  pid {pid}: {line}")

    if args.dry_run:
        return 0

    timeout = float(args.timeout)

    def pid_exists(pid: int) -> bool:
        try:
            if not utils.psutil.pid_exists(pid):
                return False
            try:
                p = utils.psutil.Process(pid)
                if p.status() == utils.psutil.STATUS_ZOMBIE:
                    return False
            except Exception:
                # If we can't query status, fall back to pid_exists.
                pass
            return True
        except Exception:
            # psutil not available or failed; fall back to kill(0)
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

    def wait_gone(pids, deadline_s: float) -> None:
        deadline = time.time() + deadline_s
        remaining = set(pids)
        while remaining and time.time() < deadline:
            for pid in list(remaining):
                if not pid_exists(pid):
                    remaining.discard(pid)
            if remaining:
                time.sleep(0.2)

    pids = [pid for pid, _ in targets]

    # Try graceful shutdown first.
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Permission denied sending SIGINT to pid {pid}", file=sys.stderr)

    # Some servers only handle SIGTERM.
    wait_gone(pids, timeout)
    remaining = [pid for pid in pids if pid_exists(pid)]
    if not remaining:
        return 0

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Permission denied sending SIGTERM to pid {pid}", file=sys.stderr)

    wait_gone(remaining, timeout)
    remaining = [pid for pid in remaining if pid_exists(pid)]
    if not remaining:
        return 0

    if not args.force:
        print("Some processes are still running. Re-run with --force to SIGKILL:")
        for pid in remaining:
            print(f"  pid {pid}")
        return 1

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Permission denied sending SIGKILL to pid {pid}", file=sys.stderr)
    wait_gone(remaining, timeout)

    remaining = [pid for pid in remaining if pid_exists(pid)]
    if remaining:
        print("Some processes could not be stopped:")
        for pid in remaining:
            print(f"  pid {pid}")
        return 1
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "stop-all":
        return cmd_stop_all(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
