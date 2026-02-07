import errno
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - runtime import check
    psutil = None

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - runtime import check
    requests = None

TOKEN_COUNTER_KEYS = [
    "llama_tokens_processed_total",
    "llama_tokens_total",
    "llama_tokens_predicted_total",
    "llama_tokens_evaluated_total",
    "llama_prompt_tokens_total",
    "llama_tokens_prompt_total",
    "llama_tokens_generated_total",
    "tokens_processed_total",
]

DEFAULT_MODEL_FILES = [
    "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
]
DEFAULT_MODEL_SUBSTRINGS = [
    "glm-4.7-flash",
]


def require(module: Any, name: str) -> None:
    if module is None:
        raise RuntimeError(
            f"Missing dependency '{name}'. Install with: pip install -r tools/llama_bench/requirements.txt"
        )


def expand_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return os.path.expandvars(os.path.expanduser(value))


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def find_run_script(path: Optional[str] = None) -> Optional[str]:
    candidate = expand_path(path) if path else expand_path("~/run-llama-server.sh")
    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    if candidate and os.path.isfile(candidate):
        return candidate
    return None


def find_llama_server(llama_cpp_dir: Optional[str] = None) -> Optional[str]:
    candidates: List[Path] = []
    path_bin = which("llama-server")
    if path_bin:
        return path_bin
    if llama_cpp_dir:
        base = Path(expand_path(llama_cpp_dir))
        candidates.extend([
            base / "llama-server",
            base / "build" / "bin" / "llama-server",
        ])
    home = Path.home()
    candidates.extend([
        home / "llama.cpp" / "llama-server",
        home / "llama.cpp" / "build" / "bin" / "llama-server",
    ])
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def find_model_path(
    model: Optional[str] = None,
    model_dir: Optional[str] = None,
    preferred_files: Optional[List[str]] = None,
    preferred_substrings: Optional[List[str]] = None,
) -> Optional[str]:
    if model:
        candidate = expand_path(model)
        if candidate and os.path.isfile(candidate):
            return candidate
        return None

    env_model = os.environ.get("MODEL")
    if env_model:
        candidate = expand_path(env_model)
        if candidate and os.path.isfile(candidate):
            return candidate

    search_dirs: List[Path] = []
    if model_dir:
        search_dirs.append(Path(expand_path(model_dir)))
    home = Path.home()
    search_dirs.extend([
        home / "models",
        Path.cwd() / "models",
    ])

    candidates: List[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        candidates.extend(sorted(directory.glob("*.gguf")))

    if not candidates:
        return None
    if preferred_files:
        for name in preferred_files:
            for candidate in candidates:
                if candidate.name == name:
                    return str(candidate)
    if preferred_substrings:
        lowered = [s.lower() for s in preferred_substrings]
        matches = [
            candidate
            for candidate in candidates
            if any(token in candidate.name.lower() for token in lowered)
        ]
        if len(matches) == 1:
            return str(matches[0])
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def list_models(model_dir: Optional[str] = None) -> List[str]:
    search_dirs: List[Path] = []
    if model_dir:
        search_dirs.append(Path(expand_path(model_dir)))
    home = Path.home()
    search_dirs.extend([
        home / "models",
        Path.cwd() / "models",
    ])
    candidates: List[str] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for item in sorted(directory.glob("*.gguf")):
            candidates.append(str(item))
    return candidates


def default_model_alias(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    return Path(model_path).stem


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def check_port_available(port: int) -> Tuple[bool, Optional[str]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        return True, None
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return False, f"Port {port} is already in use."
        return False, f"Unable to bind port {port}: {exc}"
    finally:
        try:
            sock.close()
        except Exception:
            pass


def find_llama_server_processes() -> List[Tuple[int, str]]:
    if psutil is None:
        require(psutil, "psutil")
    results: List[Tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        haystack = " ".join([name] + cmdline).lower()
        if "llama-server" in haystack:
            results.append((proc.info["pid"], " ".join(cmdline)))
    return results


def check_vram_available(min_free_mb: float) -> Tuple[bool, str]:
    gpus = get_gpu_metrics()
    if not gpus:
        return False, "No GPU metrics available (nvidia-smi/rocm-smi not found)."
    free_list: List[float] = []
    details: List[str] = []
    for gpu in gpus:
        if gpu.mem_total_mb is None or gpu.mem_used_mb is None:
            continue
        free = gpu.mem_total_mb - gpu.mem_used_mb
        free_list.append(free)
        details.append(
            f"GPU {gpu.index} {gpu.name}: free {free:.0f}MB / {gpu.mem_total_mb:.0f}MB"
        )
    if not free_list:
        return False, "GPU memory metrics unavailable."
    max_free = max(free_list)
    if max_free < min_free_mb:
        return False, f"Insufficient free VRAM. Max free {max_free:.0f}MB < {min_free_mb:.0f}MB. " + "; ".join(details)
    return True, "; ".join(details)


@dataclass
class GpuMetric:
    index: int
    name: str
    pstate: Optional[str]
    clock_graphics_mhz: Optional[float]
    clock_graphics_max_mhz: Optional[float]
    util: Optional[float]
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    power_w: Optional[float]
    power_limit_w: Optional[float]
    temp_c: Optional[float]


@dataclass
class Sample:
    ts: float
    cpu_percent: Optional[float]
    mem_percent: Optional[float]
    mem_used_mb: Optional[float]
    load1: Optional[float]
    process_cpu_percent: Optional[float]
    process_rss_mb: Optional[float]
    gpus: List[GpuMetric]
    tokens_total: Optional[float]
    rapl_cpu_energy_uj: Optional[int]
    rapl_sys_energy_uj: Optional[int]


@dataclass
class SampleSummary:
    duration_s: float
    avg_cpu_percent: Optional[float]
    avg_mem_percent: Optional[float]
    avg_mem_used_mb: Optional[float]
    avg_process_cpu_percent: Optional[float]
    avg_process_rss_mb: Optional[float]
    avg_gpu_util: Optional[float]
    avg_gpu_power_w: Optional[float]
    avg_gpu_temp_c: Optional[float]
    # GPU "normalized" signals (best-effort)
    avg_gpu_power_limit_w: Optional[float] = None
    avg_gpu_power_percent_of_limit: Optional[float] = None
    avg_gpu_mem_used_mb: Optional[float] = None
    avg_gpu_mem_total_mb: Optional[float] = None
    tokens_per_s: Optional[float] = None
    tokens_total_delta: Optional[float] = None
    tokens_total_end: Optional[float] = None
    tokens_per_s_per_w: Optional[float] = None

    # Energy/power (best-effort)
    cpu_energy_j: Optional[float] = None
    cpu_energy_wh: Optional[float] = None
    cpu_power_w: Optional[float] = None
    gpu_energy_j: Optional[float] = None
    gpu_energy_wh: Optional[float] = None
    pc_energy_j: Optional[float] = None
    pc_energy_wh: Optional[float] = None
    pc_power_w: Optional[float] = None
    pc_energy_source: Optional[str] = None


@dataclass
class RaplDomain:
    path: str
    name: str
    max_energy_range_uj: Optional[int]


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _read_int(path: str) -> Optional[int]:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except Exception:
        return None


def discover_rapl_domains() -> List[RaplDomain]:
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    domains: List[RaplDomain] = []
    try:
        for energy_path in base.glob("**/energy_uj"):
            dom_dir = energy_path.parent
            name = _read_text(str(dom_dir / "name")) or dom_dir.name
            max_range = _read_int(str(dom_dir / "max_energy_range_uj"))
            domains.append(RaplDomain(path=str(dom_dir), name=name, max_energy_range_uj=max_range))
    except Exception:
        return []
    return domains


def pick_rapl_domain(domains: List[RaplDomain], prefer_tokens: List[str]) -> Optional[RaplDomain]:
    if not domains:
        return None
    lowered = [t.lower() for t in prefer_tokens]
    for token in lowered:
        for dom in domains:
            if token in (dom.name or "").lower():
                return dom
    return None


def read_rapl_energy_uj(domain: Optional[RaplDomain]) -> Optional[int]:
    if not domain:
        return None
    return _read_int(os.path.join(domain.path, "energy_uj"))


def delta_counter(v0: Optional[int], v1: Optional[int], max_range: Optional[int]) -> Optional[int]:
    if v0 is None or v1 is None:
        return None
    if v1 >= v0:
        return int(v1 - v0)
    if max_range and max_range > 0:
        # wrapped
        return int((max_range - v0) + v1)
    return int(v1 - v0)


class MetricsSampler:
    def __init__(
        self,
        interval: float = 1.0,
        pid: Optional[int] = None,
        metrics_url: Optional[str] = None,
        include_gpu: bool = True,
    ) -> None:
        self.interval = interval
        self.pid = pid
        self.metrics_url = metrics_url
        self.include_gpu = include_gpu
        self.samples: List[Sample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._token_metric_key: Optional[str] = None
        self.on_sample = None

        # RAPL energy counters (best-effort, Linux only).
        rapl = discover_rapl_domains()
        self._rapl_cpu = pick_rapl_domain(rapl, ["package-0", "package", "cpu", "core"])
        self._rapl_sys = pick_rapl_domain(rapl, ["psys", "platform", "system"])

        if psutil is None:
            require(psutil, "psutil")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._prime_cpu()
        # Capture an initial sample immediately so short runs still
        # have a baseline for energy counters.
        try:
            sample0 = self._collect_sample()
            if sample0:
                self.samples.append(sample0)
                try:
                    if self.on_sample:
                        self.on_sample(sample0)
                except Exception:
                    pass
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
        # Capture a final sample for end-of-window energy deltas.
        try:
            sample1 = self._collect_sample()
            if sample1:
                self.samples.append(sample1)
                try:
                    if self.on_sample:
                        self.on_sample(sample1)
                except Exception:
                    pass
        except Exception:
            pass

    def _prime_cpu(self) -> None:
        if psutil is None:
            return
        psutil.cpu_percent(interval=None)
        if self.pid:
            try:
                psutil.Process(self.pid).cpu_percent(interval=None)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._collect_sample()
            if sample:
                self.samples.append(sample)
                try:
                    if self.on_sample:
                        self.on_sample(sample)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def _collect_sample(self) -> Optional[Sample]:
        if psutil is None:
            return None
        ts = time.time()
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_used_mb = mem.used / (1024 * 1024)
        load1 = None
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            load1 = None

        process_cpu = None
        process_rss = None
        if self.pid:
            try:
                proc = psutil.Process(self.pid)
                process_cpu = proc.cpu_percent(interval=None)
                process_rss = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                process_cpu = None
                process_rss = None

        gpus: List[GpuMetric] = []
        if self.include_gpu:
            gpus = get_gpu_metrics()

        tokens_total = None
        if self.metrics_url:
            tokens_total = self._fetch_tokens_total()

        rapl_cpu = read_rapl_energy_uj(self._rapl_cpu)
        rapl_sys = read_rapl_energy_uj(self._rapl_sys)

        return Sample(
            ts=ts,
            cpu_percent=cpu_percent,
            mem_percent=mem_percent,
            mem_used_mb=mem_used_mb,
            load1=load1,
            process_cpu_percent=process_cpu,
            process_rss_mb=process_rss,
            gpus=gpus,
            tokens_total=tokens_total,
            rapl_cpu_energy_uj=rapl_cpu,
            rapl_sys_energy_uj=rapl_sys,
        )

    def _fetch_tokens_total(self) -> Optional[float]:
        if requests is None:
            require(requests, "requests")
        if not self.metrics_url:
            return None
        try:
            resp = requests.get(self.metrics_url, timeout=1)
            if resp.status_code != 200:
                return None
            metrics = parse_prometheus_metrics(resp.text)
            if not metrics:
                return None
            if not self._token_metric_key:
                self._token_metric_key = choose_token_metric(metrics)
            if self._token_metric_key:
                return metrics.get(self._token_metric_key)
        except Exception:
            return None
        return None

    def summarize(self) -> SampleSummary:
        return summarize_samples(
            self.samples,
            rapl_cpu_max_range_uj=(self._rapl_cpu.max_energy_range_uj if self._rapl_cpu else None),
            rapl_sys_max_range_uj=(self._rapl_sys.max_energy_range_uj if self._rapl_sys else None),
        )

    def summarize_window(self, window_s: Optional[float]) -> SampleSummary:
        """Summarize samples over the last `window_s` seconds.

        If `window_s` is None, summarizes over all collected samples.
        """
        return summarize_samples(
            self.samples,
            window_s=window_s,
            rapl_cpu_max_range_uj=(self._rapl_cpu.max_energy_range_uj if self._rapl_cpu else None),
            rapl_sys_max_range_uj=(self._rapl_sys.max_energy_range_uj if self._rapl_sys else None),
        )


def summarize_samples(
    samples: List[Sample],
    window_s: Optional[float] = None,
    rapl_cpu_max_range_uj: Optional[int] = None,
    rapl_sys_max_range_uj: Optional[int] = None,
) -> SampleSummary:
    if not samples:
        return SampleSummary(
            duration_s=0.0,
            avg_cpu_percent=None,
            avg_mem_percent=None,
            avg_mem_used_mb=None,
            avg_process_cpu_percent=None,
            avg_process_rss_mb=None,
            avg_gpu_util=None,
            avg_gpu_power_w=None,
            avg_gpu_temp_c=None,
            avg_gpu_mem_used_mb=None,
            avg_gpu_mem_total_mb=None,
            tokens_per_s=None,
            tokens_total_delta=None,
            tokens_total_end=None,
            tokens_per_s_per_w=None,
        )

    data = samples
    if window_s is not None:
        end_ts = samples[-1].ts
        start_ts = end_ts - float(window_s)
        data = [s for s in samples if s.ts >= start_ts]
        if not data:
            data = [samples[-1]]

    duration = 0.0
    if len(data) >= 2:
        duration = max(0.001, data[-1].ts - data[0].ts)

    avg_cpu = avg_or_none([s.cpu_percent for s in data])
    avg_mem = avg_or_none([s.mem_percent for s in data])
    avg_mem_used = avg_or_none([s.mem_used_mb for s in data])
    avg_proc_cpu = avg_or_none([s.process_cpu_percent for s in data])
    avg_proc_rss = avg_or_none([s.process_rss_mb for s in data])

    gpu_utils: List[float] = []
    gpu_powers: List[float] = []
    gpu_power_limits: List[float] = []
    gpu_power_pcts: List[float] = []
    gpu_temps: List[float] = []
    gpu_mem_used: List[float] = []
    gpu_mem_total: List[float] = []
    for sample in data:
        for gpu in sample.gpus:
            if gpu.util is not None:
                gpu_utils.append(gpu.util)
            if gpu.power_w is not None:
                gpu_powers.append(gpu.power_w)
            if gpu.power_limit_w is not None:
                gpu_power_limits.append(gpu.power_limit_w)
                if gpu.power_w is not None and gpu.power_limit_w > 0:
                    try:
                        gpu_power_pcts.append(100.0 * float(gpu.power_w) / float(gpu.power_limit_w))
                    except Exception:
                        pass
            if gpu.temp_c is not None:
                gpu_temps.append(gpu.temp_c)
            if gpu.mem_used_mb is not None:
                gpu_mem_used.append(gpu.mem_used_mb)
            if gpu.mem_total_mb is not None:
                gpu_mem_total.append(gpu.mem_total_mb)

    avg_gpu_util = avg_or_none(gpu_utils)
    avg_gpu_power = avg_or_none(gpu_powers)
    avg_gpu_temp = avg_or_none(gpu_temps)
    avg_gpu_power_limit = avg_or_none(gpu_power_limits)
    avg_gpu_power_pct = avg_or_none(gpu_power_pcts)
    avg_gpu_mem_used = avg_or_none(gpu_mem_used)
    avg_gpu_mem_total = avg_or_none(gpu_mem_total)

    tokens_per_s = None
    tokens_total_delta = None
    tokens_total_end = None
    token_samples = [(s.ts, s.tokens_total) for s in data if s.tokens_total is not None]
    if token_samples:
        tokens_total_end = token_samples[-1][1]
    if len(token_samples) >= 2:
        t0, v0 = token_samples[0]
        t1, v1 = token_samples[-1]
        if v0 is not None and v1 is not None and t1 > t0:
            tokens_total_delta = (v1 - v0)
            tokens_per_s = tokens_total_delta / (t1 - t0)
    tokens_per_s_per_w = None
    if tokens_per_s and avg_gpu_power and avg_gpu_power > 0:
        tokens_per_s_per_w = tokens_per_s / avg_gpu_power

    # Energy/power (best-effort)
    cpu_energy_uj = None
    sys_energy_uj = None
    if len(data) >= 2:
        cpu_samples = [s.rapl_cpu_energy_uj for s in data if s.rapl_cpu_energy_uj is not None]
        if len(cpu_samples) >= 2:
            cpu_energy_uj = delta_counter(cpu_samples[0], cpu_samples[-1], rapl_cpu_max_range_uj)
        sys_samples = [s.rapl_sys_energy_uj for s in data if s.rapl_sys_energy_uj is not None]
        if len(sys_samples) >= 2:
            sys_energy_uj = delta_counter(sys_samples[0], sys_samples[-1], rapl_sys_max_range_uj)

    cpu_energy_j = (cpu_energy_uj / 1_000_000.0) if cpu_energy_uj is not None else None
    cpu_energy_wh = (cpu_energy_j / 3600.0) if cpu_energy_j is not None else None
    cpu_power_w = (cpu_energy_j / duration) if (cpu_energy_j is not None and duration > 0) else None

    gpu_energy_j = None
    gpu_energy_wh = None
    if avg_gpu_power is not None and duration > 0:
        gpu_energy_j = float(avg_gpu_power) * float(duration)
        gpu_energy_wh = gpu_energy_j / 3600.0

    pc_energy_j = None
    pc_energy_wh = None
    pc_power_w = None
    pc_source = None
    if sys_energy_uj is not None:
        pc_energy_j = sys_energy_uj / 1_000_000.0
        pc_energy_wh = pc_energy_j / 3600.0
        pc_power_w = (pc_energy_j / duration) if duration > 0 else None
        pc_source = "rapl_psys"
    else:
        # Estimated "whole PC" from CPU package + GPU.
        est = 0.0
        have_any = False
        if cpu_energy_j is not None:
            est += float(cpu_energy_j)
            have_any = True
        if gpu_energy_j is not None:
            est += float(gpu_energy_j)
            have_any = True
        if have_any:
            pc_energy_j = est
            pc_energy_wh = est / 3600.0
            pc_source = "estimated"
            if duration > 0:
                pc_power_w = est / duration

    return SampleSummary(
        duration_s=duration,
        avg_cpu_percent=avg_cpu,
        avg_mem_percent=avg_mem,
        avg_mem_used_mb=avg_mem_used,
        avg_process_cpu_percent=avg_proc_cpu,
        avg_process_rss_mb=avg_proc_rss,
        avg_gpu_util=avg_gpu_util,
        avg_gpu_power_w=avg_gpu_power,
        avg_gpu_temp_c=avg_gpu_temp,
        avg_gpu_power_limit_w=avg_gpu_power_limit,
        avg_gpu_power_percent_of_limit=avg_gpu_power_pct,
        avg_gpu_mem_used_mb=avg_gpu_mem_used,
        avg_gpu_mem_total_mb=avg_gpu_mem_total,
        tokens_per_s=tokens_per_s,
        tokens_total_delta=tokens_total_delta,
        tokens_total_end=tokens_total_end,
        tokens_per_s_per_w=tokens_per_s_per_w,
        cpu_energy_j=cpu_energy_j,
        cpu_energy_wh=cpu_energy_wh,
        cpu_power_w=cpu_power_w,
        gpu_energy_j=gpu_energy_j,
        gpu_energy_wh=gpu_energy_wh,
        pc_energy_j=pc_energy_j,
        pc_energy_wh=pc_energy_wh,
        pc_power_w=pc_power_w,
        pc_energy_source=pc_source,
    )


def avg_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    data = [v for v in values if v is not None]
    if not data:
        return None
    return sum(data) / len(data)


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def choose_token_metric(metrics: Dict[str, float]) -> Optional[str]:
    for key in TOKEN_COUNTER_KEYS:
        if key in metrics:
            return key
    for key in metrics:
        if "token" in key and key.endswith("_total"):
            return key
    return None


def get_gpu_metrics() -> List[GpuMetric]:
    if which("nvidia-smi"):
        return _get_nvidia_metrics()
    if which("rocm-smi"):
        return _get_rocm_metrics()
    return []


def _get_nvidia_metrics() -> List[GpuMetric]:
    # Try an extended query first (includes power limit + clocks).
    query_full = (
        "index,name,pstate,clocks.current.graphics,clocks.max.graphics,"
        "utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu"
    )
    cmd_full = [
        "nvidia-smi",
        f"--query-gpu={query_full}",
        "--format=csv,noheader,nounits",
    ]
    output = ""
    try:
        output = subprocess.check_output(cmd_full, text=True).strip()
    except Exception:
        output = ""

    if not output:
        # Fallback to the minimal query used previously.
        query = (
            "index,name,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu"
        )
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return []

    gpus: List[GpuMetric] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]

        # Full format: 11 fields
        if len(parts) >= 11:
            index = safe_int(parts[0])
            name = parts[1]
            pstate = parts[2] or None
            clk = safe_float(parts[3])
            clk_max = safe_float(parts[4])
            util = safe_float(parts[5])
            mem_used = safe_float(parts[6])
            mem_total = safe_float(parts[7])
            power = safe_float(parts[8])
            power_limit = safe_float(parts[9])
            temp = safe_float(parts[10])
        # Minimal format: 8 fields
        elif len(parts) >= 8:
            index = safe_int(parts[0])
            name = parts[1]
            pstate = None
            clk = None
            clk_max = None
            util = safe_float(parts[2])
            mem_used = safe_float(parts[4])
            mem_total = safe_float(parts[5])
            power = safe_float(parts[6])
            power_limit = None
            temp = safe_float(parts[7])
        else:
            continue

        gpus.append(
            GpuMetric(
                index=index if index is not None else 0,
                name=name,
                pstate=pstate,
                clock_graphics_mhz=clk,
                clock_graphics_max_mhz=clk_max,
                util=util,
                mem_used_mb=mem_used,
                mem_total_mb=mem_total,
                power_w=power,
                power_limit_w=power_limit,
                temp_c=temp,
            )
        )
    return gpus


def _get_rocm_metrics() -> List[GpuMetric]:
    cmd = ["rocm-smi", "--showuse", "--showpower", "--showtemp", "--json"]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return []
    try:
        data = json.loads(output)
    except Exception:
        return []
    gpus: List[GpuMetric] = []
    for key, info in data.items():
        if not key.lower().startswith("card"):
            continue
        index = safe_int(re.sub(r"\D+", "", key))
        name = str(info.get("Card series", "AMD"))
        util = safe_float(info.get("GPU use (%)"))
        mem_used = safe_float(info.get("VRAM Used Memory (B)"))
        if mem_used is not None:
            mem_used = mem_used / (1024 * 1024)
        mem_total = safe_float(info.get("VRAM Total Memory (B)"))
        if mem_total is not None:
            mem_total = mem_total / (1024 * 1024)
        power = safe_float(info.get("Average Graphics Package Power (W)"))
        temp = safe_float(info.get("Temperature (Sensor edge) (C)"))
        gpus.append(
            GpuMetric(
                index=index if index is not None else 0,
                name=name,
                pstate=None,
                clock_graphics_mhz=None,
                clock_graphics_max_mhz=None,
                util=util,
                mem_used_mb=mem_used,
                mem_total_mb=mem_total,
                power_w=power,
                power_limit_w=None,
                temp_c=temp,
            )
        )
    return gpus


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def find_pid_by_port(port: int) -> Optional[int]:
    if psutil is None:
        require(psutil, "psutil")
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return conn.pid
    except Exception:
        return None
    return None


def format_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def format_number(value: Optional[float], unit: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{unit}"


def format_sample_line(sample: Sample) -> str:
    cpu = format_percent(sample.cpu_percent)
    mem = format_percent(sample.mem_percent)
    load = format_number(sample.load1, precision=2)
    proc_cpu = format_percent(sample.process_cpu_percent)
    proc_rss = format_number(sample.process_rss_mb, "MB", precision=0)
    gpu_util = "n/a"
    gpu_power = "n/a"
    gpu_cap = "n/a"
    if sample.gpus:
        util_vals = [g.util for g in sample.gpus if g.util is not None]
        power_vals = [g.power_w for g in sample.gpus if g.power_w is not None]
        cap_vals: List[float] = []
        for g in sample.gpus:
            if g.power_w is not None and g.power_limit_w is not None and g.power_limit_w > 0:
                try:
                    cap_vals.append(100.0 * float(g.power_w) / float(g.power_limit_w))
                except Exception:
                    pass
        gpu_util = format_percent(avg_or_none(util_vals))
        gpu_power = format_number(avg_or_none(power_vals), "W", precision=1)
        gpu_cap = format_percent(avg_or_none(cap_vals))
    tokens = format_number(sample.tokens_total, precision=0)
    tokens_per_w = "n/a"
    if sample.gpus:
        power_vals = [g.power_w for g in sample.gpus if g.power_w is not None and g.power_w > 0]
        if power_vals and sample.tokens_total is not None:
            avg_power = avg_or_none(power_vals)
            if avg_power and avg_power > 0:
                tokens_per_w = format_number(sample.tokens_total / avg_power, precision=2)
    return (
        f"cpu {cpu} mem {mem} load1 {load} "
        f"proc_cpu {proc_cpu} proc_rss {proc_rss} "
        f"gpu {gpu_util} gpu_pwr {gpu_power} gpu_cap {gpu_cap} tokens {tokens} tok_per_w {tokens_per_w}"
    )


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m{rem:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes}m"
