# Agent Guide (paq_llamacpp_studio)

Python + shell toolkit around `llama.cpp`:
- `llama-run`: run `llama-server` with live stats (TUI or plain)
- `llama-bench`: benchmark prompts and aggregate tokens/sec (+ optional power/Wh)

Rules files:
- Cursor: none found in `.cursor/rules/` or `.cursorrules`
- Copilot: none found in `.github/copilot-instructions.md`


## Build / Lint / Test (copy/paste)

One-time setup (minimal runtime deps):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/llama_bench/requirements.txt
```

Dev/test tooling (matches CI intent):

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install pytest pytest-cov httpx black flake8 mypy
```

Build `llama.cpp` (clones external repo; output typically `build/llama.cpp/build/bin/llama-server`):

```bash
./scripts/build_llama.cpp.sh
```

Lint/format/typecheck:

```bash
black tools/ tests/
black --check tools/ tests/
flake8 tools/ tests/ --max-line-length=120 --extend-ignore=E203,W503
mypy tools/ --ignore-missing-imports --no-error-summary
```

Tests (pytest runs `unittest`-style tests; acceptance uses fake server `tests/bin/llama-server`):

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/acceptance/ -v

# Single file
python3 -m pytest tests/acceptance/test_e2e_acceptance.py -v

# Single test (node id)
python3 -m pytest tests/acceptance/test_e2e_acceptance.py::E2EAcceptanceTests::test_llama_run_plain_mode_prints_stats_and_exits_cleanly -v

# Single test by keyword
python3 -m pytest tests/acceptance/test_e2e_acceptance.py -k tokens_per_s -v
```


## Notes (agent-relevant)

- Acceptance tests run against a stub server; they should not require a real `llama.cpp` build, a model download, or GPU access.
- Default dev port is `11433`; OpenAI-compatible readiness is `GET /v1/models` (port open alone is not sufficient).
- Prometheus metrics are typically served at `GET /metrics` and are used for tokens/sec computations.
- `./scripts/setup.sh` may prompt to download a large model and may generate local files like `.env` and `scripts/start_server.sh`.
- Avoid committing secrets or machine-local artifacts (e.g. `.env`, model paths, large benchmark outputs).


## Run the CLIs

The repo provides thin shell wrappers that prefer `.venv/bin/python`:

```bash
./llama-run --help
./llama-bench --help

./llama-run run --port 11433
./llama-run run --ui plain --interval 0.5 --stats-interval 2
./llama-run stop-all

./llama-bench benchmark --endpoint http://127.0.0.1:11433 --prompt-sizes 1k,10k --max-output-tokens 256
./llama-bench bench --host 127.0.0.1 --port 11433 --prompt-sizes 100,200 --max-output-tokens 16
```


## Code Style (follow existing patterns)

Imports:
- Group/order: standard library, third-party, local.
- Keep blocks stable; prefer explicit imports.
- Types: prefer `Optional[T]`, `List[T]`, etc. (avoid `T | None` unless file already uses it).

Formatting:
- Black is authoritative; target 120 columns.
- Use f-strings; avoid manual alignment.
- When reading/writing text/JSON, use `encoding="utf-8"`.

Types/data modeling:
- Add type hints for public functions and non-trivial helpers.
- Use `@dataclass` for record-like structures (see `tools/llama_bench/utils.py`, `tools/llama_run/tui.py`).
- Prefer `None` for missing values over sentinel strings.

Optional deps:
- Some deps are best-effort (`psutil`, `requests`). Keep imports guarded and fail clearly via `utils.require()` when needed.

Naming:
- Files/modules: `snake_case.py`
- Functions/vars: `snake_case`
- Classes: `CapWords`
- Constants: `UPPER_SNAKE_CASE`
- CLI flags: kebab-case (argparse uses `--ctx-size`, `--max-output-tokens`, ...)

Error handling + UX:
- For CLI failures: print a user-facing message to `stderr` and return non-zero.
- Use specific exceptions; use broad `except Exception:` only for best-effort sampling, retries/backoff, or cleanup.
- Network calls must have timeouts.
- Server readiness: prefer probing `/v1/models` and/or `/metrics` (see `wait_for_server_ready`).
- Process reuse: default to refusing to start a second server unless `--allow-multi-server` is set.

Concurrency:
- Threads are used for log pumping and sampling; keep them daemonized where appropriate.
- Protect shared state with `threading.Lock`.
- Stop paths should join threads and close servers/sockets cleanly.

Shell:
- Use `set -euo pipefail`.
- Quote variables (especially paths).
- Prefer SIGINT/SIGTERM before SIGKILL.


## Repo Map

- `tools/llama_bench/`: server/monitor/benchmark/autotune (`python -m tools.llama_bench.cli ...`)
- `tools/llama_run/`: TUI/plain runner + token-meter proxy
- `tests/bin/llama-server`: wrapper around `tests/fake_llama_server.py` for acceptance tests
- `scripts/`: setup/build/download/benchmark helpers (may create `.env` and generated start scripts)


## Change Hygiene

- Keep changes small and focused; update tests when behavior changes.
- If you add CLI flags, update `--help` and keep defaults consistent across `llama-run` and `llama-bench`.
- Avoid adding heavy runtime dependencies unless necessary.
