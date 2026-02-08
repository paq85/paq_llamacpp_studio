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


def _extract_tokens(payload: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Extract (prompt_tokens, completion_tokens) from response payload."""
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or payload.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or payload.get("completion_tokens")

    # llama.cpp alternative fields
    if prompt_tokens is None:
        prompt_tokens = payload.get("prompt_eval_count") or payload.get("prompt_eval_tokens")
    if completion_tokens is None:
        completion_tokens = payload.get("eval_count") or payload.get("tokens_predicted")

    if prompt_tokens is None or completion_tokens is None:
        return None
    try:
        return (int(prompt_tokens), int(completion_tokens))
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    lock: threading.Lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self.lock:
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self.total_tokens = self.prompt_tokens + self.completion_tokens
            self.last_prompt_tokens = int(prompt_tokens)
            self.last_completion_tokens = int(completion_tokens)

    def get(self) -> int:
        with self.lock:
            return int(self.total_tokens)

    def get_prompt(self) -> int:
        with self.lock:
            return int(self.prompt_tokens)

    def get_completion(self) -> int:
        with self.lock:
            return int(self.completion_tokens)

    def get_last_prompt(self) -> int:
        with self.lock:
            return int(self.last_prompt_tokens)

    def get_last_completion(self) -> int:
        with self.lock:
            return int(self.last_completion_tokens)

    def get_total_used(self) -> int:
        """Get total tokens used in current session (prompt + completion)."""
        with self.lock:
            return int(self.total_tokens)

    def get_current_context(self) -> int:
        """Get total tokens used in current request (prompt + completion)."""
        with self.lock:
            return int(self.last_prompt_tokens + self.last_completion_tokens)


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

    def _is_streaming_response(self) -> bool:
        """Check if the response will be streamed (Transfer-Encoding: chunked)."""
        # llama.cpp typically uses chunked encoding for streaming
        return self.headers.get("Transfer-Encoding", "").lower() == "chunked"

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

        # Read request body (buffered). This keeps proxy simple and works for bench + common clients.
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

        # Process streaming response if applicable
        if self._is_streaming_response() and self.command == "POST" and resp.status_code == 200:
            if self.path.startswith("/v1/chat/completions") or self.path.startswith("/v1/completions"):
                self._process_streaming_response(resp)
            else:
                # Non-completion streaming endpoint
                content_bytes = resp.content or b""
                self._send_response(resp, content_bytes)
        else:
            # Non-streaming response
            content_bytes = resp.content or b""

            # Token metering (best-effort, non-stream JSON only)
            if self.command == "POST" and resp.status_code == 200:
                if self.path.startswith("/v1/chat/completions") or self.path.startswith("/v1/completions"):
                    payload = _read_json_bytes(content_bytes)
                    if payload:
                        tokens = _extract_tokens(payload)
                        if tokens is not None:
                            prompt_tokens, completion_tokens = tokens
                            self.meter.add(prompt_tokens, completion_tokens)

            # Send response for non-streaming
            self._send_response(resp, content_bytes)

    def _send_response(self, resp, content_bytes: bytes) -> None:
        """Send HTTP response to client."""
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
        self.send_header("Content-Length", str(len(content_bytes)))
        self.end_headers()
        if content_bytes:
            self.wfile.write(content_bytes)

    def _process_streaming_response(self, resp) -> None:
        """Process streaming response and extract tokens from final message."""
        content_parts = []
        prompt_tokens = None
        completion_tokens = None
        has_seen_data_prefix = False

        try:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    content_parts.append(chunk.decode("utf-8", errors="ignore"))
                    self.wfile.write(chunk)

                    # Try to parse SSE (Server-Sent Events) chunks
                    chunk_text = chunk.decode("utf-8", errors="ignore")

                    # Check if this chunk starts a new SSE message
                    if chunk_text.startswith("data: "):
                        # Split by "data: " to get individual messages
                        messages = chunk_text.split("data: ")[1:]

                        for msg in messages:
                            msg = msg.strip()
                            if not msg or msg == "[DONE]":
                                continue

                            try:
                                payload = json.loads(msg)
                                if payload.get("choices") and len(payload["choices"]) > 0:
                                    choice = payload["choices"][0]
                                    if "message" in choice:
                                        # This might be a chat completion
                                        continue
                                    if "delta" in choice:
                                        # This is a streaming chunk
                                        delta = choice["delta"]
                                        if "content" in delta:
                                            # We're still streaming, continue collecting
                                            continue

                                    # Check for prompt_tokens in usage
                                    if "usage" in choice and "prompt_tokens" in choice["usage"]:
                                        prompt_tokens = choice["usage"]["prompt_tokens"]

                                    # Check for completion_tokens in usage
                                    if "usage" in choice and "completion_tokens" in choice["usage"]:
                                        completion_tokens = choice["usage"]["completion_tokens"]
                            except Exception:
                                # Not a valid JSON chunk, skip
                                continue

            # After streaming completes, parse the full content if tokens not found
            if prompt_tokens is None or completion_tokens is None:
                full_content = "".join(content_parts)
                if full_content:
                    payload = _read_json_bytes(full_content.encode("utf-8"))
                    if payload:
                        tokens = _extract_tokens(payload)
                        if tokens is not None:
                            prompt_tokens, completion_tokens = tokens

            # Add tokens to meter if found
            if prompt_tokens is not None and completion_tokens is not None:
                self.meter.add(prompt_tokens, completion_tokens)

        except Exception:
            # If streaming processing fails, try to parse the complete response
            try:
                content_bytes = resp.content or b""
                if content_bytes:
                    payload = _read_json_bytes(content_bytes)
                    if payload:
                        tokens = _extract_tokens(payload)
                        if tokens is not None:
                            prompt_tokens, completion_tokens = tokens
                            self.meter.add(prompt_tokens, completion_tokens)
            except Exception:
                pass


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
