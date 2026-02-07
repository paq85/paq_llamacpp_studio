import curses
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from tools.llama_bench import utils
from tools.llama_run.logperf import LogPerfMeter, TokenWindow


@dataclass
class TuiState:
    title: str
    endpoint: str
    pid: int
    started_at: float
    sampler: utils.MetricsSampler
    perf: LogPerfMeter
    logs: Deque[str]
    show_logs: bool = False
    paused: bool = False
    last_error: Optional[str] = None


def _clamp(n: int, lo: int, hi: int) -> int:
    return lo if n < lo else hi if n > hi else n


def _bar(value: Optional[float], width: int, lo: float = 0.0, hi: float = 100.0) -> str:
    if width <= 0:
        return ""
    if value is None:
        return " " * width
    v = float(value)
    v = lo if v < lo else hi if v > hi else v
    filled = int(round((v - lo) / (hi - lo) * width)) if hi > lo else 0
    filled = _clamp(filled, 0, width)
    return "#" * filled + "-" * (width - filled)


def _safe_add(stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        h, w = stdscr.getmaxyx()
        if y < 0 or y >= h:
            return
        if x < 0 or x >= w:
            return
        stdscr.addnstr(y, x, text, max(0, w - x - 1), attr)
    except Exception:
        return


def _fmt_tok_total(summary: utils.SampleSummary) -> str:
    if summary.tokens_total_end is None:
        return "n/a"
    return f"{summary.tokens_total_end:,.0f}"


def _fmt_tps(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.1f}"


def _fmt_tok(v: int) -> str:
    return f"{int(v):,d}"


def _fmt_tok_rate(summary: utils.SampleSummary) -> str:
    if summary.tokens_per_s is None:
        return "n/a"
    return f"{summary.tokens_per_s:,.2f}"


def _fmt_tok_delta(summary: utils.SampleSummary) -> str:
    if summary.tokens_total_delta is None:
        return "n/a"
    return f"{summary.tokens_total_delta:,.0f}"


def _fmt_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:5.1f}%"


def _fmt_num(v: Optional[float], unit: str = "", width: int = 0, prec: int = 1) -> str:
    if v is None:
        s = "n/a"
    else:
        s = f"{v:.{prec}f}{unit}"
    if width:
        return s.rjust(width)
    return s


def _fmt_rss_mb(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.0f}MB"


def _fmt_vram(summary: utils.SampleSummary) -> str:
    if summary.avg_gpu_mem_used_mb is None or summary.avg_gpu_mem_total_mb is None:
        return "n/a"
    return f"{summary.avg_gpu_mem_used_mb/1024.0:.1f}/{summary.avg_gpu_mem_total_mb/1024.0:.1f}GB"


def _fmt_w(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.1f}W"


def _fmt_wh(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    wh = float(v)
    if wh >= 1000.0:
        return f"{wh/1000.0:,.2f}kWh"
    return f"{wh:,.2f}Wh"


def _draw(stdscr, state: TuiState) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    now = time.time()
    uptime_s = max(0.0, now - state.started_at)
    uptime = utils.human_duration(uptime_s)

    all_sum = state.sampler.summarize_window(None)
    now_sum = state.sampler.summarize_window(5.0)
    m1 = state.sampler.summarize_window(60.0)
    m15 = state.sampler.summarize_window(15.0 * 60.0)

    perf_all = state.perf.window(None)
    perf_now = state.perf.last()
    perf_1m = state.perf.window(60.0)
    perf_15m = state.perf.window(15.0 * 60.0)

    header = f"{state.title}  pid {state.pid}  {state.endpoint}  up {uptime}"
    if state.paused:
        header += "  [PAUSED]"
    _safe_add(stdscr, 0, 0, header[: max(0, w - 1)], curses.A_BOLD)

    help_line = "q quit | l logs | p pause"
    _safe_add(stdscr, 1, 0, help_line)

    p_total, g_total = state.perf.totals()
    _safe_add(
        stdscr,
        2,
        0,
        f"tokens processed: prompt {_fmt_tok(p_total)} | gen {_fmt_tok(g_total)} | total {_fmt_tok(p_total + g_total)}",
    )

    # Window table (performance)
    _safe_add(
        stdscr,
        4,
        0,
        "window   p_tok/s   g_tok/s   p_tok     g_tok     cpu     mem     rss        gpu    gcap     gpuW     vram",
        curses.A_UNDERLINE,
    )

    def row(y: int, label: str, s: utils.SampleSummary, p: TokenWindow) -> None:
        line = (
            f"{label:<6} "
            f"{_fmt_tps(p.prompt_tps):>8} "
            f"{_fmt_tps(p.gen_tps):>8} "
            f"{_fmt_tok(p.prompt_tokens):>9} "
            f"{_fmt_tok(p.gen_tokens):>9} "
            f"{_fmt_pct(s.avg_cpu_percent):>7} "
            f"{_fmt_pct(s.avg_mem_percent):>7} "
            f"{_fmt_rss_mb(s.avg_process_rss_mb):>10} "
            f"{_fmt_pct(s.avg_gpu_util):>7} "
            f"{_fmt_pct(s.avg_gpu_power_percent_of_limit):>7} "
            f"{_fmt_num(s.avg_gpu_power_w, 'W', width=7, prec=1)} "
            f"{_fmt_vram(s):>10}"
        )
        _safe_add(stdscr, y, 0, line)

    row(5, "now", now_sum, perf_now)
    row(6, "1m", m1, perf_1m)
    row(7, "15m", m15, perf_15m)
    row(8, "all", all_sum, perf_all)

    # Bars (now window, few seconds)
    bar_w = max(10, min(40, w - 20))
    cpu_bar = _bar(now_sum.avg_cpu_percent, bar_w)
    mem_bar = _bar(now_sum.avg_mem_percent, bar_w)
    gpu_bar = _bar(now_sum.avg_gpu_util, bar_w)

    gpu_cap_bar = _bar(now_sum.avg_gpu_power_percent_of_limit, bar_w)

    _safe_add(stdscr, 10, 0, f"cpu  [{cpu_bar}] {_fmt_pct(now_sum.avg_cpu_percent)}")
    _safe_add(stdscr, 11, 0, f"mem  [{mem_bar}] {_fmt_pct(now_sum.avg_mem_percent)}")
    _safe_add(stdscr, 12, 0, f"gpu  [{gpu_bar}] {_fmt_pct(now_sum.avg_gpu_util)}")
    _safe_add(stdscr, 13, 0, f"gcap [{gpu_cap_bar}] {_fmt_pct(now_sum.avg_gpu_power_percent_of_limit)}")

    # Energy/power (window)
    pc_tag = "pc" if all_sum.pc_energy_source != "estimated" else "pc*"
    _safe_add(
        stdscr,
        15,
        0,
        f"energy (Wh) cpu/gpu/{pc_tag} | power (W) cpu/{pc_tag}",
        curses.A_UNDERLINE,
    )

    def erow(y: int, label: str, s: utils.SampleSummary) -> None:
        line = (
            f"{label:<6} "
            f"{_fmt_wh(s.cpu_energy_wh):>10} "
            f"{_fmt_wh(s.gpu_energy_wh):>10} "
            f"{_fmt_wh(s.pc_energy_wh):>10} "
            f"| {_fmt_w(s.cpu_power_w):>9} "
            f"{_fmt_w(s.pc_power_w):>9}"
        )
        # (pc label is indicated by * in the header and values; nothing else needed)
        _safe_add(stdscr, y, 0, line)

    erow(16, "now", now_sum)
    erow(17, "1m", m1)
    erow(18, "15m", m15)
    erow(19, "all", all_sum)

    if state.last_error:
        _safe_add(stdscr, 21, 0, f"error: {state.last_error}"[: max(0, w - 1)], curses.A_BOLD)

    # Logs pane (tail)
    if state.show_logs:
        start_y = 23
        if start_y < h - 1:
            _safe_add(stdscr, start_y, 0, "logs (tail):", curses.A_UNDERLINE)
            max_lines = max(0, h - (start_y + 2))
            tail = list(state.logs)[-max_lines:]
            for i, line in enumerate(tail):
                _safe_add(stdscr, start_y + 1 + i, 0, line.rstrip("\n")[: max(0, w - 1)])

    stdscr.refresh()


def run_tui(state: TuiState, stop_cb, alive_cb) -> int:
    """Blocking curses UI loop.

    stop_cb() is invoked when the user requests exit.
    """

    def _main(stdscr) -> int:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(200)

        while True:
            if not alive_cb():
                state.last_error = state.last_error or "server exited"
                _draw(stdscr, state)
                time.sleep(0.5)
                return 0
            try:
                ch = stdscr.getch()
            except Exception:
                ch = -1

            if ch != -1:
                if ch in (ord("q"), 27):
                    stop_cb()
                    return 0
                if ch == ord("l"):
                    state.show_logs = not state.show_logs
                if ch == ord("p"):
                    state.paused = not state.paused

            if not state.paused:
                _draw(stdscr, state)

    return curses.wrapper(_main)
