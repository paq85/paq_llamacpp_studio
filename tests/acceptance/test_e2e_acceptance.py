import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(host: str, port: int, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


class E2EAcceptanceTests(unittest.TestCase):
    def test_llama_run_plain_mode_reports_tokens_per_s_numeric(self) -> None:
        port = _find_free_port()
        dummy_model = Path(tempfile.mkdtemp()) / "dummy.gguf"
        dummy_model.write_bytes(b"not a real gguf\n")

        cmd = [
            str(REPO_ROOT / "llama-run"),
            "run",
            "--ui",
            "plain",
            "--llama-server",
            str(REPO_ROOT / "tests/bin/llama-server"),
            "--model",
            str(dummy_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-gpu",
            "--interval",
            "0.2",
            "--stats-interval",
            "1",
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertTrue(_wait_port("127.0.0.1", port, timeout_s=5.0))
            # Generate some token traffic through the public endpoint.
            import requests

            # Wait for server ready behind proxy.
            deadline = time.time() + 10.0
            while time.time() < deadline:
                try:
                    rr = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                    if rr.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(0.1)
            else:
                self.fail("server did not become ready for /v1/models")

            for _i in range(4):
                r = requests.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hello" * 200}],
                        "max_tokens": 16,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    timeout=5,
                )
                self.assertEqual(r.status_code, 200)
                time.sleep(0.2)

            # Wait long enough to collect >1 metrics sample and print at least 1 stats block.
            time.sleep(2.5)
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()

        # Find the last "now" token line.
        now_lines = [line for line in out.splitlines() if line.startswith("[tok] now:")]
        self.assertTrue(now_lines, msg=f"no token now lines found. stdout:\n{out}\nstderr:\n{err}")
        last = now_lines[-1]

        m_p = re.search(r"p_tok/s\s+([0-9]+(?:\.[0-9]+)?)", last)
        m_g = re.search(r"g_tok/s\s+([0-9]+(?:\.[0-9]+)?)", last)
        self.assertTrue(m_p or m_g, msg=f"could not parse tok/s from line: {last}")
        if m_p:
            self.assertGreater(float(m_p.group(1)), 0.0)
        if m_g:
            self.assertGreater(float(m_g.group(1)), 0.0)

        m_pt = re.search(r"p_tok\s+(\d+)", last)
        m_gt = re.search(r"g_tok\s+(\d+)", last)
        self.assertIsNotNone(m_pt)
        self.assertIsNotNone(m_gt)
        assert m_pt is not None and m_gt is not None
        self.assertGreater(int(m_pt.group(1)) + int(m_gt.group(1)), 0)

        self.assertEqual(proc.returncode, 0, msg=f"stderr was:\n{err}")

    def test_llama_run_plain_mode_prints_stats_and_exits_cleanly(self) -> None:
        port = _find_free_port()
        dummy_model = Path(tempfile.mkdtemp()) / "dummy.gguf"
        dummy_model.write_bytes(b"not a real gguf\n")

        cmd = [
            str(REPO_ROOT / "llama-run"),
            "run",
            "--ui",
            "plain",
            "--llama-server",
            str(REPO_ROOT / "tests/bin/llama-server"),
            "--model",
            str(dummy_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-gpu",
            "--interval",
            "0.2",
            "--stats-interval",
            "1",
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertTrue(_wait_port("127.0.0.1", port, timeout_s=5.0))
            time.sleep(2.5)
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()

        # It should print periodic sys + token stats in plain mode.
        self.assertIn("[sys] now:", out)
        self.assertIn("[sys] 1m:", out)
        self.assertIn("[sys] 15m:", out)
        self.assertIn("[sys] all:", out)
        self.assertIn("[tok] now:", out)
        self.assertIn("[tok] 1m:", out)
        self.assertIn("[tok] 15m:", out)
        self.assertIn("[tok] all:", out)
        self.assertIn("llama-server exited with code", out)
        # No hard requirement for energy availability; just ensure the formatter ran.
        self.assertIn("energy cpu", out)
        self.assertEqual(proc.returncode, 0, msg=f"stderr was:\n{err}")

    def test_llama_bench_bench_reuses_existing_server_and_does_not_stop_it(self) -> None:
        port = _find_free_port()
        dummy_model = Path(tempfile.mkdtemp()) / "dummy.gguf"
        dummy_model.write_bytes(b"not a real gguf\n")

        server_cmd = [
            str(REPO_ROOT / "tests/bin/llama-server"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        server = subprocess.Popen(
            server_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(_wait_port("127.0.0.1", port, timeout_s=5.0))

            json_out = Path(tempfile.mkdtemp()) / "bench.json"
            bench_cmd = [
                str(REPO_ROOT / "llama-bench"),
                "bench",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--skip-vram-check",
                "--prompt-sizes",
                "100,200",
                "--max-output-tokens",
                "5",
                "--temperature",
                "0.0",
                "--json-out",
                str(json_out),
            ]
            res = subprocess.run(
                bench_cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            self.assertEqual(res.returncode, 0, msg=f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
            self.assertIn("Reusing existing llama-server", res.stdout)
            self.assertTrue(json_out.exists())
            # Server should still be alive and port open.
            self.assertIsNone(server.poll())
            self.assertTrue(_port_open("127.0.0.1", port))
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=10)
                except Exception:
                    server.kill()

    def test_llama_bench_bench_starts_and_stops_server_when_none_running(self) -> None:
        port = _find_free_port()
        dummy_model = Path(tempfile.mkdtemp()) / "dummy.gguf"
        dummy_model.write_bytes(b"not a real gguf\n")

        json_out = Path(tempfile.mkdtemp()) / "bench.json"
        bench_cmd = [
            str(REPO_ROOT / "llama-bench"),
            "bench",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--allow-multi-server",
            "--llama-server",
            str(REPO_ROOT / "tests/bin/llama-server"),
            "--model",
            str(dummy_model),
            "--skip-vram-check",
            "--startup-timeout",
            "10",
            "--prompt-sizes",
            "100",
            "--max-output-tokens",
            "1",
            "--temperature",
            "0.0",
            "--json-out",
            str(json_out),
        ]
        res = subprocess.run(
            bench_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        self.assertEqual(res.returncode, 0, msg=f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
        self.assertIn("Starting llama-server", res.stdout)
        self.assertTrue(json_out.exists())

        # It should have stopped the server (port closed shortly after).
        deadline = time.time() + 5.0
        while time.time() < deadline and _port_open("127.0.0.1", port):
            time.sleep(0.1)
        self.assertFalse(_port_open("127.0.0.1", port), msg="expected port to be closed")


if __name__ == "__main__":
    unittest.main()
