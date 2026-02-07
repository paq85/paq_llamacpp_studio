import socket
import subprocess
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


class StopAllAcceptanceTests(unittest.TestCase):
    def test_stop_all_stops_external_llama_server(self) -> None:
        port = _find_free_port()
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

            res = subprocess.run(
                [str(REPO_ROOT / "llama-run"), "stop-all"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            self.assertIn("Found llama-server processes", res.stdout)
            self.assertEqual(res.returncode, 0, msg=f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")

            # Port should close shortly after.
            deadline = time.time() + 5.0
            while time.time() < deadline and _port_open("127.0.0.1", port):
                time.sleep(0.1)
            self.assertFalse(_port_open("127.0.0.1", port))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except Exception:
                    server.kill()


if __name__ == "__main__":
    unittest.main()
