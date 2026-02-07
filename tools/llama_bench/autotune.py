import json
import signal
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from . import benchmark
from . import utils


@dataclass
class ServerConfig:
    llama_server: str
    model_path: str
    model_alias: Optional[str]
    host: str
    port: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    cache_type_k: str
    cache_type_v: str
    fit: str
    flash_attn: str
    kv_unified: str
    jinja: str
    chat_template_kwargs: Optional[str]
    temp: float
    top_p: float
    min_p: float
    extra_args: List[str]


@dataclass
class AutotuneResult:
    score: float
    objective: str
    tokens_per_s: Optional[float]
    tokens_per_s_per_w: Optional[float]
    duration_s: float
    avg_gpu_power_w: Optional[float]
    avg_gpu_util: Optional[float]
    avg_cpu_percent: Optional[float]
    avg_mem_percent: Optional[float]
    avg_process_cpu_percent: Optional[float]
    avg_process_rss_mb: Optional[float]
    config: Dict[str, Any]


@dataclass
class RunResult:
    ok: bool
    error: Optional[str]
    benchmark: Optional[benchmark.BenchmarkResult]


def parse_int_list(value: str) -> List[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in parts]


def build_server_cmd(config: ServerConfig) -> List[str]:
    cmd = [
        config.llama_server,
        "--model",
        config.model_path,
        "--port",
        str(config.port),
        "--fit",
        config.fit,
        "--temp",
        str(config.temp),
        "--top-p",
        str(config.top_p),
        "--min-p",
        str(config.min_p),
        "--ctx-size",
        str(config.ctx_size),
        "--batch-size",
        str(config.batch_size),
        "--ubatch-size",
        str(config.ubatch_size),
        "--cache-type-k",
        config.cache_type_k,
        "--cache-type-v",
        config.cache_type_v,
        "--flash-attn",
        config.flash_attn,
    ]
    if config.model_alias:
        cmd.extend(["--alias", config.model_alias])
    if config.host:
        cmd.extend(["--host", config.host])
    if config.kv_unified == "on":
        cmd.append("--kv-unified")
    if config.jinja == "on":
        cmd.append("--jinja")
        if config.chat_template_kwargs:
            cmd.extend(["--chat-template-kwargs", config.chat_template_kwargs])
    cmd.extend(config.extra_args)
    return cmd


def evaluate_candidate(
    config: ServerConfig,
    prompt_tokens: int,
    max_output_tokens: int,
    temperature: float,
    mode: str,
    model_name: Optional[str],
    extra_request: Optional[Dict[str, Any]],
    interval: float,
    startup_timeout: float,
) -> RunResult:
    cmd = build_server_cmd(config)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not utils.wait_for_port(config.host, config.port, timeout_s=startup_timeout):
            return RunResult(ok=False, error="server did not start", benchmark=None)

        endpoint = f"http://{config.host}:{config.port}"
        metrics_url = f"{endpoint}/metrics"
        sampler = utils.MetricsSampler(interval=interval, pid=proc.pid, metrics_url=metrics_url)
        result = benchmark.run_single_benchmark(
            endpoint=endpoint,
            task="list",
            prompt_tokens_target=prompt_tokens,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            mode=mode,
            model=model_name,
            extra=extra_request,
            sampler=sampler,
        )
        return RunResult(ok=True, error=None, benchmark=result)
    except Exception as exc:
        return RunResult(ok=False, error=str(exc), benchmark=None)
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def score_result(result: benchmark.BenchmarkResult, objective: str) -> float:
    if objective == "efficiency":
        if result.tokens_per_s_per_w is not None:
            return result.tokens_per_s_per_w
        if result.tokens_per_s is not None:
            return result.tokens_per_s
        return 0.0
    if result.tokens_per_s is not None:
        return result.tokens_per_s
    return 0.0


def format_candidate(config: ServerConfig) -> str:
    return f"batch={config.batch_size} ubatch={config.ubatch_size} flash={config.flash_attn} ctx={config.ctx_size}"


def build_candidates(
    base: ServerConfig,
    batch_sizes: List[int],
    ubatch_sizes: List[int],
) -> List[ServerConfig]:
    candidates: List[ServerConfig] = []
    for batch in batch_sizes:
        for ubatch in ubatch_sizes:
            if ubatch > batch:
                continue
            candidate = ServerConfig(**{**asdict(base), "batch_size": batch, "ubatch_size": ubatch})
            candidates.append(candidate)
    return candidates


def write_best(path: str, result: Optional[AutotuneResult]) -> None:
    if not result:
        return
    payload = {"best": asdict(result)}
    try:
        config = ServerConfig(**result.config)
        payload["server_cmd"] = build_server_cmd(config)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_autotune(
    llama_cpp_dir: Optional[str],
    llama_server: Optional[str],
    model: Optional[str],
    model_dir: Optional[str],
    host: str,
    ctx_size: int,
    batch_sizes: List[int],
    ubatch_sizes: List[int],
    duration_s: float,
    prompt_tokens: int,
    max_output_tokens: int,
    temperature: float,
    mode: str,
    model_name: Optional[str],
    objective: str,
    interval: float,
    startup_timeout: float,
    cache_type_k: str,
    cache_type_v: str,
    fit: str,
    flash_attn: str,
    kv_unified: str,
    jinja: str,
    chat_template_kwargs: Optional[str],
    extra_request: Optional[Dict[str, Any]],
    extra_args: List[str],
    output_path: str,
    min_vram_free_mb: float,
    skip_vram_check: bool,
    allow_multi_server: bool,
) -> Tuple[Optional[AutotuneResult], List[AutotuneResult]]:
    llama_bin = llama_server or utils.find_llama_server(llama_cpp_dir)
    if not llama_bin:
        raise RuntimeError("llama-server not found. Provide --llama-server or --llama-cpp-dir.")

    model_path = utils.find_model_path(
        model,
        model_dir,
        preferred_files=utils.DEFAULT_MODEL_FILES,
        preferred_substrings=utils.DEFAULT_MODEL_SUBSTRINGS,
    )
    if not model_path:
        raise RuntimeError(
            "Model not found. Provide --model or place GLM-4.7-Flash .gguf in ~/models."
        )

    if not allow_multi_server:
        running = utils.find_llama_server_processes()
        if running:
            raise RuntimeError("Detected running llama-server processes. Stop them before autotune.")

    if not skip_vram_check:
        ok, reason = utils.check_vram_available(min_vram_free_mb)
        if not ok:
            raise RuntimeError(reason)

    base = ServerConfig(
        llama_server=llama_bin,
        model_path=model_path,
        model_alias=utils.default_model_alias(model_path),
        host=host,
        port=0,
        ctx_size=ctx_size,
        batch_size=batch_sizes[0],
        ubatch_size=ubatch_sizes[0],
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        fit=fit,
        flash_attn=flash_attn,
        kv_unified=kv_unified,
        jinja=jinja,
        chat_template_kwargs=chat_template_kwargs,
        temp=temperature,
        top_p=0.95,
        min_p=0.01,
        extra_args=extra_args,
    )
    if model_name is None:
        model_name = base.model_alias

    candidates = build_candidates(base, batch_sizes, ubatch_sizes)
    if not candidates:
        raise RuntimeError("No candidates to test. Check batch/ubatch sizes.")

    best: Optional[AutotuneResult] = None
    results: List[AutotuneResult] = []
    deadline = time.time() + duration_s

    total = len(candidates)
    try:
        for idx, candidate in enumerate(candidates, 1):
            if time.time() >= deadline:
                print("Time limit reached. Stopping autotune.")
                break

            if not skip_vram_check:
                ok, reason = utils.check_vram_available(min_vram_free_mb)
                if not ok:
                    print(reason)
                    break

            candidate.port = utils.find_free_port()
            print(f"[{idx}/{total}] Testing {format_candidate(candidate)}")
            run = evaluate_candidate(
                config=candidate,
                prompt_tokens=prompt_tokens,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                mode=mode,
                model_name=model_name,
                extra_request=extra_request,
                interval=interval,
                startup_timeout=startup_timeout,
            )
            if not run.ok or not run.benchmark:
                print(f"  Failed: {run.error or 'unknown error'}")
                continue

            score = score_result(run.benchmark, objective)
            result = AutotuneResult(
                score=score,
                objective=objective,
                tokens_per_s=run.benchmark.tokens_per_s,
                tokens_per_s_per_w=run.benchmark.tokens_per_s_per_w,
                duration_s=run.benchmark.duration_s,
                avg_gpu_power_w=run.benchmark.avg_gpu_power_w,
                avg_gpu_util=run.benchmark.avg_gpu_util,
                avg_cpu_percent=run.benchmark.avg_cpu_percent,
                avg_mem_percent=run.benchmark.avg_mem_percent,
                avg_process_cpu_percent=run.benchmark.avg_process_cpu_percent,
                avg_process_rss_mb=run.benchmark.avg_process_rss_mb,
                config=asdict(candidate),
            )
            results.append(result)

            tokens_line = "n/a" if result.tokens_per_s is None else f"{result.tokens_per_s:.2f} tok/s"
            eff_line = (
                "n/a" if result.tokens_per_s_per_w is None else f"{result.tokens_per_s_per_w:.4f} tok/s/W"
            )
            print(f"  Result: {tokens_line} | {eff_line} | score {score:.4f}")
            if result.tokens_per_s_per_w is not None:
                print(f"    tokens/s/W: {result.tokens_per_s_per_w:.4f}")

            if best is None or result.score > best.score:
                best = result
                print("  New best found.")
                write_best(output_path, best)
    except KeyboardInterrupt:
        print("Autotune interrupted. Saving best settings so far.")
    finally:
        if best:
            write_best(output_path, best)

    return best, results
