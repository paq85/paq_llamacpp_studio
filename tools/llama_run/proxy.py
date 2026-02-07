import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from tools.llama_bench import utils


class ReusableTCPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


def _is_hop_header(name: str) -> bool:
    n = name.lower()
    return n in {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }


def _read_json_bytes(data: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _extract_total_tokens(payload: Dict[str, Any]) -> Optional[int]:
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or payload.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or payload.get("completion_tokens")
    total_tokens = usage.get("total_tokens") or payload.get("total_tokens")

    # llama.cpp alternative fields
    if prompt_tokens is None:
        prompt_tokens = payload.get("prompt_eval_count") or payload.get("prompt_eval_tokens")
    if completion_tokens is None:
        completion_tokens = payload.get("eval_count") or payload.get("tokens_predicted")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    if total_tokens is None:
        return None
    try:
        return int(total_tokens)
    except Exception:
        return None


@dataclass
class TokenMeter:
    total_tokens: int = 0
    lock: threading.Lock = threading.Lock()

    def add(self, n: int) -> None:
        with self.lock:
            self.total_tokens += int(n)

    def get(self) -> int:
        with self.lock:
            return int(self.total_tokens)


class ProxyHandler(BaseHTTPRequestHandler):
    # ThreadingHTTPServer

    server_version = "llama-run-proxy"

    def log_message(self, format: str, *args) -> None:
        # Keep quiet; llama-run UI already shows enough.
        return

    @property
    def backend(self) -> Tuple[str, int]:
        return self.server.backend  # type: ignore[attr-defined]

    @property
    def meter(self) -> TokenMeter:
        return self.server.meter  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def _handle(self) -> None:
        # Provide our own metrics endpoint so tools can always compute tok/s.
        if self.path == "/metrics":
            total = self.meter.get()
            body = "".join(
                [
                    "# HELP llama_tokens_processed_total Total tokens processed (metered by llama-run)\n",
                    "# TYPE llama_tokens_processed_total counter\n",
                    f"llama_tokens_processed_total {total}\n",
                ]
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        backend_host, backend_port = self.backend
        url = f"http://{backend_host}:{backend_port}{self.path}"

        # Read body (buffered). This keeps proxy simple and works for bench + common clients.
        length = 0
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""

        headers: Dict[str, str] = {}
        for k, v in self.headers.items():
            if _is_hop_header(k) or k.lower() == "host":
                continue
            headers[k] = v

        utils.require(utils.requests, "requests")
        try:
            resp = utils.requests.request(
                self.command,
                url,
                headers=headers,
                data=body,
                allow_redirects=False,
                timeout=600,
            )
        except Exception as exc:
            # Common during startup while the backend is still binding/initializing.
            data = f"upstream not ready: {exc}\n".encode("utf-8")
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        content = resp.content or b""

        # Token metering (best-effort, non-stream JSON only)
        if self.command == "POST" and resp.status_code == 200:
            if self.path.startswith("/v1/chat/completions") or self.path.startswith("/v1/completions"):
                payload = _read_json_bytes(content)
                if payload:
                    total = _extract_total_tokens(payload)
                    if total is not None and total >= 0:
                        self.meter.add(total)

        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if _is_hop_header(k):
                continue
            # requests may decompress but keep header; safest is to strip content-encoding
            if k.lower() == "content-encoding":
                continue
            if k.lower() == "content-length":
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if content:
            self.wfile.write(content)


class LlamaMeterProxy:
    def __init__(self, host: str, port: int, backend_host: str, backend_port: int) -> None:
        self.host = host
        self.port = port
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.meter = TokenMeter()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        httpd = ReusableTCPServer((self.host, int(self.port)), ProxyHandler)
        httpd.backend = (self.backend_host, int(self.backend_port))  # type: ignore[attr-defined]
        httpd.meter = self.meter  # type: ignore[attr-defined]
        self._httpd = httpd

        def _serve() -> None:
            try:
                httpd.serve_forever(poll_interval=0.2)
            except Exception:
                return

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        time.sleep(0.2)
