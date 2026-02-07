# llama-bench

CLI helper to run llama.cpp server, monitor resources, and benchmark prompt lengths.

## Setup

```bash
pip install -r tools/llama_bench/requirements.txt
```

## Run server (direct llama-server)

```bash
python3 -m tools.llama_bench.cli server --monitor --model ~/models/your-model.gguf
```

If `~/models/GLM-4.7-Flash-UD-Q4_K_XL.gguf` exists, you can omit `--model`.

The server will refuse to start if another `llama-server` is running or if free VRAM is below the default minimum (8192MB). Use `--allow-multi-server` or `--skip-vram-check` to override.

## Monitor an existing server

```bash
python3 -m tools.llama_bench.cli monitor --port 11433 --interval 1
```

## Benchmark prompt sizes

```bash
python3 -m tools.llama_bench.cli benchmark --endpoint http://127.0.0.1:11433 \
  --prompt-sizes 1k,10k,100k --max-output-tokens 1024 --port 11433
```

Tasks:
- `--task coding` (default): realistic coding task
- `--task list`: simple long list generation (good for pushing output tokens)

To run a more realistic prompt, use `--task coding`:

```bash
./llama-bench benchmark --endpoint http://127.0.0.1:11433 \
  --task coding --prompt-sizes 1k,10k --max-output-tokens 1024
```

To test multiple output lengths, use `--output-tokens` (cross-product with `--prompt-sizes`):

```bash
./llama-bench benchmark --endpoint http://127.0.0.1:11433 \
  --prompt-sizes 1k,10k \
  --output-tokens 16,256,1024
```

Use `--json-out` to capture results to a file.

If you want periodic feedback while each request runs (helpful for very long prompts):

```bash
./llama-bench benchmark --endpoint http://127.0.0.1:11433 \
  --prompt-sizes 1k,10k,100k \
  --progress on --progress-interval 5
```

## One-shot benchmark (reuse or start server)

If you want a single command that reuses an already-running `llama-server` when possible,
or starts one if needed:

```bash
./llama-bench bench --prompt-sizes 1k,10k,100k --json-out bench.json
```

If you want the server to stay up after the benchmark finishes:

```bash
./llama-bench bench --keep-server
```

## Autotune server settings (default 10 minutes)

```bash
python3 -m tools.llama_bench.cli autotune
```

Autotune saves the best settings to `tools/llama_bench/tuned.json`. You can stop anytime with Ctrl+C and it will keep the best found so far.
