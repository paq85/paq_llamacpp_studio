# llama-run

Simple CLI to run `llama-server` with a lightweight, htop-like TUI.

It reuses the sampling/metrics helpers from `tools/llama_bench`.

## Setup

This repo is not packaged, so run it via `python -m`.

If you already installed deps for `llama-bench`, you're set. Otherwise:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/llama_bench/requirements.txt
```

## Run

```bash
./llama-run run --port 11433
```

## Stop All Servers

Stop all running `llama-server` processes (even if they were started outside `llama-run`):

```bash
./llama-run stop-all
```

Preview what would be stopped:

```bash
./llama-run stop-all --dry-run
```

Keys (TUI):
- `q` quit (sends SIGINT to server)
- `l` toggle log tail pane
- `p` pause screen updates

What it shows:
- Tokens processed + throughput (parsed from llama.cpp log lines like `prompt eval time = ... tokens per second`)
- CPU/RAM, process RSS
- GPU util/power/VRAM (NVIDIA via `nvidia-smi`, AMD via `rocm-smi`)
- CPU package power/energy and whole-PC energy when available via RAPL (`/sys/class/powercap`)
  - If a platform/psys counter isn't available, whole-PC energy is estimated as CPU(package)+GPU.

Window semantics:
- `now`: last reported prompt/gen speeds from llama.cpp (updates when a request completes)
- `1m` / `15m`: aggregated over requests completed in that window
- `all`: aggregated since startup

Pass extra `llama-server` args after `--`:

```bash
./llama-run run -- --n-gpu-layers 999
```

If you want the old scrolling output:

```bash
./llama-run run --ui plain
```
