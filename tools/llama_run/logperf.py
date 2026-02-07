import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


_RE_PROMPT = re.compile(
    r"prompt\s+eval\s+time\s*=\s*(?P<ms>[0-9.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens.*?(?P<tps>[0-9.]+)\s*tokens\s+per\s+second",
    re.IGNORECASE,
)
_RE_EVAL = re.compile(
    r"(^|\s)eval\s+time\s*=\s*(?P<ms>[0-9.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens.*?(?P<tps>[0-9.]+)\s*tokens\s+per\s+second",
    re.IGNORECASE,
)


@dataclass
class TokenWindow:
    duration_s: float
    prompt_tokens: int
    gen_tokens: int
    prompt_tps: Optional[float]
    gen_tps: Optional[float]

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.gen_tokens)


@dataclass
class _TokenEvent:
    ts: float
    kind: str  # 'prompt' or 'gen'
    tokens: int
    duration_s: float
    tps: Optional[float]


class LogPerfMeter:
    """Extract token counts + timings from llama.cpp server logs.

    llama-server prints lines like:
      prompt eval time =  537.88 ms /  744 tokens ... 1383.21 tokens per second
             eval time = 1861.34 ms /  256 tokens ...  137.54 tokens per second
    """

    def __init__(self, retention_s: float = 16 * 60.0) -> None:
        self.retention_s = float(retention_s)
        self._lock = threading.Lock()
        self._events: Deque[_TokenEvent] = deque()
        self._prompt_total_tokens = 0
        self._gen_total_tokens = 0
        self._prompt_total_dur_s = 0.0
        self._gen_total_dur_s = 0.0

        self._last_prompt: Optional[_TokenEvent] = None
        self._last_gen: Optional[_TokenEvent] = None

    def totals(self) -> Tuple[int, int]:
        with self._lock:
            return int(self._prompt_total_tokens), int(self._gen_total_tokens)

    def add_log_line(self, line: str, ts: Optional[float] = None) -> None:
        if not line:
            return
        now = time.time() if ts is None else float(ts)
        m = _RE_PROMPT.search(line)
        if m:
            self._add_event(
                _TokenEvent(
                    ts=now,
                    kind="prompt",
                    tokens=int(m.group("tokens")),
                    duration_s=float(m.group("ms")) / 1000.0,
                    tps=float(m.group("tps")),
                )
            )
            return
        m = _RE_EVAL.search(line)
        if m:
            self._add_event(
                _TokenEvent(
                    ts=now,
                    kind="gen",
                    tokens=int(m.group("tokens")),
                    duration_s=float(m.group("ms")) / 1000.0,
                    tps=float(m.group("tps")),
                )
            )
            return

    def _add_event(self, ev: _TokenEvent) -> None:
        with self._lock:
            self._events.append(ev)
            if ev.kind == "prompt":
                self._prompt_total_tokens += int(ev.tokens)
                self._prompt_total_dur_s += float(ev.duration_s)
                self._last_prompt = ev
            else:
                self._gen_total_tokens += int(ev.tokens)
                self._gen_total_dur_s += float(ev.duration_s)
                self._last_gen = ev

            self._prune_locked(now_ts=ev.ts)

    def _prune_locked(self, now_ts: float) -> None:
        cutoff = float(now_ts) - self.retention_s
        while self._events and self._events[0].ts < cutoff:
            self._events.popleft()

    def window(self, window_s: Optional[float]) -> TokenWindow:
        with self._lock:
            if window_s is None:
                p_tok = int(self._prompt_total_tokens)
                g_tok = int(self._gen_total_tokens)
                p_dur = float(self._prompt_total_dur_s)
                g_dur = float(self._gen_total_dur_s)
            else:
                now = time.time()
                cutoff = now - float(window_s)
                p_tok = 0
                g_tok = 0
                p_dur = 0.0
                g_dur = 0.0
                # Iterate from newest to oldest.
                for ev in reversed(self._events):
                    if ev.ts < cutoff:
                        break
                    if ev.kind == "prompt":
                        p_tok += int(ev.tokens)
                        p_dur += float(ev.duration_s)
                    else:
                        g_tok += int(ev.tokens)
                        g_dur += float(ev.duration_s)

        p_tps = (p_tok / p_dur) if p_dur > 0 else None
        g_tps = (g_tok / g_dur) if g_dur > 0 else None
        dur = 0.0
        if p_dur > 0 or g_dur > 0:
            dur = p_dur + g_dur
        return TokenWindow(
            duration_s=dur,
            prompt_tokens=p_tok,
            gen_tokens=g_tok,
            prompt_tps=p_tps,
            gen_tps=g_tps,
        )

    def last(self) -> TokenWindow:
        """Return the most recently observed prompt/gen perf sample.

        This is *not* a time window; it's the last reported values from llama.cpp.
        Useful as a "right now" approximation that doesn't decay to n/a when idle.
        """
        with self._lock:
            p = self._last_prompt
            g = self._last_gen

        p_tok = int(p.tokens) if p else 0
        g_tok = int(g.tokens) if g else 0
        p_dur = float(p.duration_s) if p else 0.0
        g_dur = float(g.duration_s) if g else 0.0

        p_tps = None
        g_tps = None
        if p and p.tps is not None:
            p_tps = float(p.tps)
        elif p_dur > 0:
            p_tps = p_tok / p_dur

        if g and g.tps is not None:
            g_tps = float(g.tps)
        elif g_dur > 0:
            g_tps = g_tok / g_dur

        return TokenWindow(
            duration_s=p_dur + g_dur,
            prompt_tokens=p_tok,
            gen_tokens=g_tok,
            prompt_tps=p_tps,
            gen_tps=g_tps,
        )
