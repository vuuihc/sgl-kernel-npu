#!/usr/bin/env python3
"""A/B benchmark for the legacy and tile-parallel apply_token_bitmask kernels."""

import argparse
import datetime
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch: int
    vocab: int
    dtype: str
    mask_mode: str
    indices_count: int = 0


QUICK_CASES = [
    BenchmarkCase("b1_v32000_fp16_sparse", 1, 32000, "float16", "sparse"),
    BenchmarkCase("b1_v128256_fp16_sparse", 1, 128256, "float16", "sparse"),
    BenchmarkCase("b1_v151936_bf16_sparse", 1, 151936, "bfloat16", "sparse"),
    BenchmarkCase("b8_v151936_bf16_sparse", 8, 151936, "bfloat16", "sparse"),
]

FULL_CASES = QUICK_CASES + [
    BenchmarkCase("b1_v200000_bf16_sparse", 1, 200000, "bfloat16", "sparse"),
    BenchmarkCase("b1_v151936_bf16_random", 1, 151936, "bfloat16", "random"),
    BenchmarkCase("b1_v151936_bf16_unmasked", 1, 151936, "bfloat16", "unmasked"),
    BenchmarkCase("b2_v151936_bf16_sparse", 2, 151936, "bfloat16", "sparse"),
    BenchmarkCase("b32_v151936_bf16_sparse", 32, 151936, "bfloat16", "sparse"),
    BenchmarkCase("b1_v32100_fp16_sparse", 1, 32100, "float16", "sparse"),
    BenchmarkCase("idx8_b16_v151936_bf16", 16, 151936, "bfloat16", "sparse", 8),
]

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

OPS = {
    "legacy": torch.ops.npu.apply_token_bitmask_legacy,
    "optimized": torch.ops.npu.apply_token_bitmask,
}


def _run(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _soc_version():
    try:
        return str(torch_npu.npu._backends.get_soc_version())
    except (AttributeError, RuntimeError):
        return None


def _metadata(args):
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status": _run(["git", "status", "--short", "--untracked-files=no"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_npu": str(getattr(torch_npu, "__version__", None)),
        "sgl_kernel_npu": str(getattr(sgl_kernel_npu, "__version__", None)),
        "device_name": torch.npu.get_device_name(),
        "soc_version": _soc_version(),
        "device_count": torch.npu.device_count(),
        "suite": args.suite,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "timing": "torch.npu.Event(enable_timing=True)",
    }


def _make_bitmask(batch, vocab, mode, generator):
    width = math.ceil(vocab / 32)
    if mode == "unmasked":
        return torch.full((batch, width), -1, dtype=torch.int32)

    density = 0.5 if mode == "random" else 0.001
    bits = torch.rand((batch, width * 32), generator=generator) < density
    weights = 1 << torch.arange(32, dtype=torch.int64)
    packed = (bits.reshape(batch, width, 32).to(torch.int64) * weights).sum(dim=-1)
    return packed.to(torch.int32)


def _make_indices(case):
    if case.indices_count == 0:
        return None
    return (
        torch.linspace(
            0,
            case.batch - 1,
            steps=case.indices_count,
            dtype=torch.float64,
        )
        .round()
        .to(torch.int32)
    )


def _reference(logits, bitmask, indices):
    words = bitmask.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, dtype=torch.int64)
    allowed = ((words[:, :, None] >> shifts) & 1).reshape(bitmask.shape[0], -1)
    allowed = allowed[:, : logits.shape[1]].bool()
    rows = (
        indices.to(torch.int64)
        if indices is not None
        else torch.arange(logits.shape[0], dtype=torch.int64)
    )
    expected = logits.clone()
    selected = expected[rows]
    selected[~allowed[rows]] = float("-inf")
    expected[rows] = selected
    return expected


def _assert_correct(case, logits, bitmask, indices):
    expected = _reference(logits, bitmask, indices)
    indices_npu = indices.npu() if indices is not None else None
    outputs = {}
    for variant, op in OPS.items():
        output = op(logits.npu(), bitmask.npu(), indices_npu).cpu()
        torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)
        outputs[variant] = output
    torch.testing.assert_close(
        outputs["optimized"], outputs["legacy"], atol=0.0, rtol=0.0
    )


def _percentile(samples, q):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _statistics(samples):
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "samples_ms": samples,
    }


def _benchmark_case(case, warmup, iterations, generator):
    dtype = DTYPES[case.dtype]
    logits = torch.randn((case.batch, case.vocab), dtype=dtype, generator=generator)
    bitmask = _make_bitmask(case.batch, case.vocab, case.mask_mode, generator)
    indices = _make_indices(case)
    _assert_correct(case, logits, bitmask, indices)

    bitmask_npu = bitmask.npu()
    indices_npu = indices.npu() if indices is not None else None
    inputs = {variant: logits.npu() for variant in OPS}

    for iteration in range(warmup):
        order = (
            ("legacy", "optimized") if iteration % 2 == 0 else ("optimized", "legacy")
        )
        for variant in order:
            OPS[variant](inputs[variant], bitmask_npu, indices_npu)
    torch.npu.synchronize()

    event_pairs = {variant: [] for variant in OPS}
    for iteration in range(iterations):
        order = (
            ("legacy", "optimized") if iteration % 2 == 0 else ("optimized", "legacy")
        )
        for variant in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            OPS[variant](inputs[variant], bitmask_npu, indices_npu)
            end.record()
            event_pairs[variant].append((start, end))
    torch.npu.synchronize()

    timings = {
        variant: [start.elapsed_time(end) for start, end in pairs]
        for variant, pairs in event_pairs.items()
    }
    stats = {variant: _statistics(samples) for variant, samples in timings.items()}
    legacy_p50 = stats["legacy"]["p50_ms"]
    optimized_p50 = stats["optimized"]["p50_ms"]
    return {
        "case": asdict(case),
        "correctness": "passed",
        "legacy": stats["legacy"],
        "optimized": stats["optimized"],
        "speedup_p50": legacy_p50 / optimized_p50,
        "latency_reduction_p50_percent": (legacy_p50 - optimized_p50)
        / legacy_p50
        * 100,
    }


def _summary(results):
    speedups = [result["speedup_p50"] for result in results]
    return {
        "geomean_speedup_p50": math.exp(
            statistics.fmean(math.log(value) for value in speedups)
        ),
        "min_speedup_p50": min(speedups),
        "max_speedup_p50": max(speedups),
        "regression_count": sum(value < 0.95 for value in speedups),
    }


def _write_markdown(path, payload):
    lines = [
        "# apply_token_bitmask A/B results",
        "",
        f"- Commit: `{payload['metadata']['git_commit']}`",
        f"- Device: `{payload['metadata']['device_name']}`",
        f"- SOC: `{payload['metadata']['soc_version']}`",
        f"- Timing: `{payload['metadata']['timing']}`",
        f"- Iterations: `{payload['metadata']['iterations']}`",
        "",
        "| Case | Legacy P50 (ms) | Optimized P50 (ms) | Speedup | P99 speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        legacy = result["legacy"]
        optimized = result["optimized"]
        p99_speedup = legacy["p99_ms"] / optimized["p99_ms"]
        lines.append(
            f"| {result['case']['name']} | {legacy['p50_ms']:.6f} | "
            f"{optimized['p50_ms']:.6f} | {result['speedup_p50']:.3f}x | "
            f"{p99_speedup:.3f}x |"
        )
    lines.extend(
        [
            "",
            f"Geomean P50 speedup: **{payload['summary']['geomean_speedup_p50']:.3f}x**",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 2:
        parser.error("--warmup must be >= 1 and --iterations must be >= 2")
    if not hasattr(torch.ops.npu, "apply_token_bitmask_legacy"):
        parser.error(
            "apply_token_bitmask_legacy is unavailable; install this experiment branch"
        )

    generator = torch.Generator().manual_seed(args.seed)
    cases = QUICK_CASES if args.suite == "quick" else FULL_CASES
    results = []
    for case in cases:
        print(f"Running {case.name}...", flush=True)
        results.append(_benchmark_case(case, args.warmup, args.iterations, generator))

    payload = {
        "schema_version": 1,
        "metadata": _metadata(args),
        "summary": _summary(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    _write_markdown(markdown_output, payload)
    print(markdown_output.read_text(encoding="utf-8"))
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_output}")


if __name__ == "__main__":
    main()
