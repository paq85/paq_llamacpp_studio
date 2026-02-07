#!/usr/bin/env python3

"""A tiny llama-server stand-in for end-to-end tests.

Implements:
- GET  /metrics
- GET  /v1/models
- POST /v1/chat/completions
- POST /v1/completions

It is intentionally minimal and only supports the subset used by tools.llama_bench
and tools.llama_run.
"""

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional


class State:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.tokens_total = 0
        self.lock = threading.Lock()
        self.alive = True

    def compute_tokens_total(self) -> int:
        # Monotonic-ish counter (tokens). Keep it simple: 1k tokens/sec baseline.
        dt = max(0.0, time.time() - self.t0)
        base = int(dt * 1000)
        with self.lock:
            return max(self.tokens_total, base)

    def bump_tokens(self, delta: int) -> None:
        with self.lock:
            self.tokens_total += int(delta)


def _read_json(handler: BaseHTTPRequestHandler) -> Optional[Dict[str, Any]]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except Exception:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _text(handler: BaseHTTPRequestHandler, status: int, text: str, content_type: str = "text/plain") -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-llama-server"

    def log_message(self, format: str, *args) -> None:
        # Keep log lines somewhat llama.cpp-ish.
        sys.stdout.write("srv  " + (format % args) + "\n")
        sys.stdout.flush()

    @property
    def state(self) -> State:
        # ThreadingHTTPServer
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/metrics":
            total = self.state.compute_tokens_total()
            body = "".join(
                [
                    "# HELP llama_tokens_processed_total Total tokens processed\n",
                    "# TYPE llama_tokens_processed_total counter\n",
                    f"llama_tokens_processed_total {total}\n",
                ]
            )
            _text(self, 200, body)
            return

        if self.path == "/v1/models":
            _json(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "fake-model",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "fake",
                        }
                    ],
                },
            )
            return

        _text(self, 404, "not found\n")

    def do_POST(self) -> None:
        data = _read_json(self)
        if data is None:
            _json(self, 400, {"error": {"message": "invalid json"}})
            return

        if self.path == "/v1/chat/completions":
            prompt = ""
            messages = data.get("messages") or []
            if messages and isinstance(messages, list):
                last = messages[-1] if messages else {}
                prompt = str((last or {}).get("content") or "")
            max_tokens = int(data.get("max_tokens") or 0)

            prompt_tokens = max(1, len(prompt) // 6)
            completion_tokens = max(0, max_tokens)
            total_tokens = prompt_tokens + completion_tokens
            self.state.bump_tokens(total_tokens)

            # Emit llama.cpp-ish perf lines for log parsing.
            # Use simple fixed speeds and compute durations from token counts.
            prompt_tps = 5000.0
            gen_tps = 200.0
            prompt_ms = (prompt_tokens / prompt_tps) * 1000.0
            gen_ms = (completion_tokens / gen_tps) * 1000.0 if completion_tokens > 0 else 0.0
            sys.stdout.write(
                f"prompt eval time = {prompt_ms:10.2f} ms / {prompt_tokens:5d} tokens (  0.00 ms per token, {prompt_tps:7.2f} tokens per second)\n"
            )
            sys.stdout.write(
                f"       eval time = {gen_ms:10.2f} ms / {completion_tokens:5d} tokens (  0.00 ms per token, {gen_tps:7.2f} tokens per second)\n"
            )
            sys.stdout.flush()

            _json(
                self,
                200,
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": data.get("model") or "fake-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                },
            )
            return

        if self.path == "/v1/completions":
            prompt = str(data.get("prompt") or "")
            max_tokens = int(data.get("max_tokens") or 0)
            prompt_tokens = max(1, len(prompt) // 6)
            completion_tokens = max(0, max_tokens)
            total_tokens = prompt_tokens + completion_tokens
            self.state.bump_tokens(total_tokens)

            prompt_tps = 5000.0
            gen_tps = 200.0
            prompt_ms = (prompt_tokens / prompt_tps) * 1000.0
            gen_ms = (completion_tokens / gen_tps) * 1000.0 if completion_tokens > 0 else 0.0
            sys.stdout.write(
                f"prompt eval time = {prompt_ms:10.2f} ms / {prompt_tokens:5d} tokens (  0.00 ms per token, {prompt_tps:7.2f} tokens per second)\n"
            )
            sys.stdout.write(
                f"       eval time = {gen_ms:10.2f} ms / {completion_tokens:5d} tokens (  0.00 ms per token, {gen_tps:7.2f} tokens per second)\n"
            )
            sys.stdout.flush()

            _json(
                self,
                200,
                {
                    "id": "cmpl-fake",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": data.get("model") or "fake-model",
                    "choices": [
                        {"index": 0, "text": "ok", "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                },
            )
            return

        _json(self, 404, {"error": {"message": "not found"}})


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11433)

    # Accept (and ignore) llama-server-ish flags.
    parser.add_argument("--model")
    parser.add_argument("--alias")
    parser.add_argument("--ctx-size")
    parser.add_argument("--batch-size")
    parser.add_argument("--ubatch-size")
    parser.add_argument("--cache-type-k")
    parser.add_argument("--cache-type-v")
    parser.add_argument("--fit")
    parser.add_argument("--flash-attn")
    parser.add_argument("--kv-unified", action="store_true")
    parser.add_argument("--jinja", action="store_true")
    parser.add_argument("--chat-template-kwargs")
    parser.add_argument("--temp")
    parser.add_argument("--top-p")
    parser.add_argument("--min-p")

    parser.add_argument("--help", action="store_true")
    args, _rest = parser.parse_known_args()

    if args.help:
        sys.stdout.write("fake llama-server (test stub)\n")
        return 0

    state = State()

    httpd = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    httpd.state = state  # type: ignore[attr-defined]

    stop = threading.Event()

    def _shutdown_worker() -> None:
        stop.wait()
        try:
            httpd.shutdown()
        except Exception:
            pass

    threading.Thread(target=_shutdown_worker, daemon=True).start()

    def _on_signal(_sig, _frame) -> None:
        # IMPORTANT: BaseServer.shutdown() must not be called from the
        # serve_forever() thread. Signals run on the main thread, which is
        # also serve_forever() here, so we signal a helper thread instead.
        state.alive = False
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    sys.stdout.write(f"main: server is listening on http://{args.host}:{args.port}\n")
    sys.stdout.flush()
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
