import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import utils


@dataclass
class BenchmarkResult:
    task: str
    prompt_target_tokens: int
    max_output_tokens: int
    prompt_chars: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    duration_s: float
    tokens_per_s: Optional[float]
    completion_tokens_per_s: Optional[float]
    avg_gpu_power_w: Optional[float]
    avg_gpu_util: Optional[float]
    avg_cpu_percent: Optional[float]
    avg_mem_percent: Optional[float]
    avg_process_cpu_percent: Optional[float]
    avg_process_rss_mb: Optional[float]
    tokens_per_s_per_w: Optional[float]
    gpu_energy_wh: Optional[float]
    cpu_energy_wh: Optional[float]
    pc_energy_wh: Optional[float]
    avg_gpu_power_limit_w: Optional[float]
    avg_gpu_power_percent_of_limit: Optional[float]


def parse_size_list(value: str) -> List[int]:
    sizes: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(parse_size_token(part))
    return sizes


def parse_size_token(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1000)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    return int(value)


def build_prompt(
    target_tokens: int,
    output_tokens: Optional[int] = None,
    base_text: Optional[str] = None,
) -> str:
    base = base_text or (
        "This is a benchmark prompt used to measure llama.cpp server performance. "
        "It should be long enough to exercise the context window. "
    )
    approx_chars = max(1, int(target_tokens * 4))
    prompt = base
    if len(prompt) < approx_chars:
        filler = (
            "\n"
            "Context filler: the next lines are repeated to reach the desired prompt size.\n"
        )
        while len(prompt) < approx_chars:
            prompt += filler
    return prompt[:approx_chars]


def build_task_prompt(task: str, prompt_tokens_target: int, max_output_tokens: int) -> str:
    task = (task or "coding").strip().lower()

    if task == "coding":
        base = (
            "You are an expert software engineer.\n"
            "\n"
            "Task: Implement a small Python library and its tests.\n"
            "Return ONLY code blocks.\n"
            "\n"
            "File 1: string_tools.py\n"
            "```python\n"
            "from __future__ import annotations\n"
            "\n"
            "import re\n"
            "from dataclasses import dataclass\n"
            "from typing import Iterable, List, Optional\n"
            "\n"
            "\n"
            "_WS_RE = re.compile(r\"\\s+\")\n"
            "\n"
            "\n"
            "def normalize_whitespace(text: str) -> str:\n"
            "    \"\"\"Collapse all whitespace to single spaces and trim.\"\"\"\n"
            "    # TODO\n"
            "    raise NotImplementedError\n"
            "\n"
            "\n"
            "def slugify(text: str, *, max_len: int = 64) -> str:\n"
            "    \"\"\"Create a URL slug: lowercase, ascii, dash-separated, no leading/trailing dashes.\"\"\"\n"
            "    # Rules:\n"
            "    # - Convert to lowercase\n"
            "    # - Replace any run of non [a-z0-9] chars with '-'\n"
            "    # - Collapse multiple '-'\n"
            "    # - Trim '-' from ends\n"
            "    # - Enforce max_len (trim, then re-trim '-')\n"
            "    # TODO\n"
            "    raise NotImplementedError\n"
            "\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class MatchSpan:\n"
            "    start: int\n"
            "    end: int\n"
            "\n"
            "\n"
            "def find_all_spans(text: str, needle: str) -> List[MatchSpan]:\n"
            "    \"\"\"Return non-overlapping spans where needle occurs in text (left-to-right).\"\"\"\n"
            "    # TODO\n"
            "    raise NotImplementedError\n"
            "```\n"
            "\n"
            "File 2: test_string_tools.py\n"
            "```python\n"
            "import unittest\n"
            "\n"
            "import string_tools as st\n"
            "\n"
            "\n"
            "class TestStringTools(unittest.TestCase):\n"
            "    def test_normalize_whitespace_basic(self):\n"
            "        self.assertEqual(st.normalize_whitespace('  a\\t b\\n  c  '), 'a b c')\n"
            "\n"
            "    def test_slugify_basic(self):\n"
            "        self.assertEqual(st.slugify('Hello, World!'), 'hello-world')\n"
            "\n"
            "    def test_find_all_spans_basic(self):\n"
            "        spans = st.find_all_spans('aaaa', 'aa')\n"
            "        self.assertEqual([(s.start, s.end) for s in spans], [(0, 2), (2, 4)])\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
            "```\n"
            "\n"
            "Requirements:\n"
            "- Must be Python 3.10+\n"
            "- No external dependencies\n"
            "- Add at least 40 tests total, covering edge cases\n"
            "\n"
            "Output format:\n"
            "- Print the full contents of both files as code blocks\n"
            "- No explanation text\n"
        )
        # Fill to desired prompt size using realistic-ish code comments.
        filler = "\n# Additional context: handle tricky unicode whitespace and punctuation.\n"
        prompt = base
        approx_chars = max(1, int(prompt_tokens_target * 4))
        # Never truncate the base coding task (it contains code blocks).
        if approx_chars < len(prompt):
            approx_chars = len(prompt)
        while len(prompt) < approx_chars:
            prompt += filler
        prompt = prompt[:approx_chars]

        # Encourage using the output budget.
        if max_output_tokens >= 512:
            prompt += (
                "\n\n"
                "Also include a section at the end of test_string_tools.py with a large table of test vectors "
                "as a Python list literal named EXTRA_VECTORS with exactly 150 entries.\n"
            )
        return prompt

    # Default / fallback task: long list.
    prompt = build_prompt(prompt_tokens_target, base_text=None)
    if max_output_tokens and max_output_tokens > 0:
        prompt += (
            "\n\n"
            "Instruction: Output a long numbered list starting at 1, one item per line, "
            "and keep going until you are cut off by the max output token limit. "
            "Do not stop early. Do not add any extra commentary.\n"
        )
    return prompt


def call_completion(
    endpoint: str,
    prompt: str,
    mode: str,
    max_tokens: int,
    temperature: float,
    model: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    utils.require(utils.requests, "requests")
    url = endpoint.rstrip("/")
    if mode == "chat":
        url = f"{url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    else:
        url = f"{url}/v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

    if model:
        payload["model"] = model
    if extra:
        payload.update(extra)

    resp = utils.requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def extract_token_usage(data: Dict[str, Any]) -> Dict[str, Optional[int]]:
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or data.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or data.get("completion_tokens")
    total_tokens = usage.get("total_tokens") or data.get("total_tokens")

    # llama.cpp alternative fields
    if prompt_tokens is None:
        prompt_tokens = data.get("prompt_eval_count") or data.get("prompt_eval_tokens")
    if completion_tokens is None:
        completion_tokens = data.get("eval_count") or data.get("tokens_predicted")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    return {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }


def run_single_benchmark(
    endpoint: str,
    task: str,
    prompt_tokens_target: int,
    max_output_tokens: int,
    temperature: float,
    mode: str,
    model: Optional[str],
    extra: Optional[Dict[str, Any]],
    sampler: Optional[utils.MetricsSampler],
) -> BenchmarkResult:
    prompt = build_task_prompt(task, prompt_tokens_target=prompt_tokens_target, max_output_tokens=max_output_tokens)
    prompt_chars = len(prompt)

    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    duration = 0.0
    response: Dict[str, Any]
    try:
        response = call_completion(
            endpoint=endpoint,
            prompt=prompt,
            mode=mode,
            max_tokens=max_output_tokens,
            temperature=temperature,
            model=model,
            extra=extra,
        )
        duration = time.perf_counter() - t0
    finally:
        if duration <= 0:
            duration = time.perf_counter() - t0
        if sampler:
            sampler.stop()

    if sampler:
        summary = sampler.summarize()
    else:
        summary = utils.SampleSummary(
            duration_s=duration,
            avg_cpu_percent=None,
            avg_mem_percent=None,
            avg_mem_used_mb=None,
            avg_process_cpu_percent=None,
            avg_process_rss_mb=None,
            avg_gpu_util=None,
            avg_gpu_power_w=None,
            avg_gpu_temp_c=None,
        )

    usage = extract_token_usage(response)
    total_tokens = usage["total_tokens"]
    completion_tokens = usage["completion_tokens"]
    tokens_per_s = None
    completion_tokens_per_s = None
    if total_tokens is not None:
        tokens_per_s = total_tokens / duration if duration > 0 else None
    if completion_tokens is not None:
        completion_tokens_per_s = completion_tokens / duration if duration > 0 else None

    tokens_per_s_per_w = None
    if summary.avg_gpu_power_w and tokens_per_s:
        tokens_per_s_per_w = tokens_per_s / summary.avg_gpu_power_w

    # Prefer request wall-time for GPU energy when sampling is too sparse.
    gpu_energy_wh = summary.gpu_energy_wh
    if gpu_energy_wh is None and summary.avg_gpu_power_w is not None and duration > 0:
        gpu_energy_wh = float(summary.avg_gpu_power_w) * float(duration) / 3600.0

    return BenchmarkResult(
        task=task,
        prompt_target_tokens=prompt_tokens_target,
        max_output_tokens=max_output_tokens,
        prompt_chars=prompt_chars,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        duration_s=duration,
        tokens_per_s=tokens_per_s,
        completion_tokens_per_s=completion_tokens_per_s,
        avg_gpu_power_w=summary.avg_gpu_power_w,
        avg_gpu_util=summary.avg_gpu_util,
        avg_cpu_percent=summary.avg_cpu_percent,
        avg_mem_percent=summary.avg_mem_percent,
        avg_process_cpu_percent=summary.avg_process_cpu_percent,
        avg_process_rss_mb=summary.avg_process_rss_mb,
        tokens_per_s_per_w=tokens_per_s_per_w,
        gpu_energy_wh=gpu_energy_wh,
        cpu_energy_wh=summary.cpu_energy_wh,
        pc_energy_wh=summary.pc_energy_wh,
        avg_gpu_power_limit_w=summary.avg_gpu_power_limit_w,
        avg_gpu_power_percent_of_limit=summary.avg_gpu_power_percent_of_limit,
    )


def run_benchmarks(
    endpoint: str,
    prompt_sizes: List[int],
    max_output_tokens: int,
    temperature: float,
    mode: str,
    model: Optional[str],
    extra: Optional[Dict[str, Any]],
    sampler_factory,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for size in prompt_sizes:
        sampler = sampler_factory() if sampler_factory else None
        result = run_single_benchmark(
            endpoint=endpoint,
            task="list",
            prompt_tokens_target=size,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            mode=mode,
            model=model,
            extra=extra,
            sampler=sampler,
        )
        results.append(result)
    return results


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    return {
        "task": result.task,
        "prompt_target_tokens": result.prompt_target_tokens,
        "max_output_tokens": result.max_output_tokens,
        "prompt_chars": result.prompt_chars,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "duration_s": result.duration_s,
        "tokens_per_s": result.tokens_per_s,
        "completion_tokens_per_s": result.completion_tokens_per_s,
        "avg_gpu_power_w": result.avg_gpu_power_w,
        "avg_gpu_util": result.avg_gpu_util,
        "avg_cpu_percent": result.avg_cpu_percent,
        "avg_mem_percent": result.avg_mem_percent,
        "avg_process_cpu_percent": result.avg_process_cpu_percent,
        "avg_process_rss_mb": result.avg_process_rss_mb,
        "tokens_per_s_per_w": result.tokens_per_s_per_w,
        "gpu_energy_wh": result.gpu_energy_wh,
        "cpu_energy_wh": result.cpu_energy_wh,
        "pc_energy_wh": result.pc_energy_wh,
        "avg_gpu_power_limit_w": result.avg_gpu_power_limit_w,
        "avg_gpu_power_percent_of_limit": result.avg_gpu_power_percent_of_limit,
    }


def results_to_json(results: List[BenchmarkResult]) -> str:
    payload = {
        "results": [result_to_dict(r) for r in results],
    }
    return json.dumps(payload, indent=2)
