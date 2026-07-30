# Ascend 950 apply_token_bitmask A/B plan

This experiment compares two kernels in the same wheel:

- `torch.ops.npu.apply_token_bitmask_legacy`: upstream row-level partitioning.
- `torch.ops.npu.apply_token_bitmask`: candidate row-by-tile partitioning.

Both paths share input validation, padding, indices handling, host allocations,
stream handling, and launch arguments. The benchmark alternates their launch
order and uses NPU timing events, so the measured difference is limited to tile
size and AIV work distribution.

## Baseline

The experiment branch is based on commit
`c3c34bafce9926ed519fbdbafaa694ccdbf9e176`, the head of upstream PR #632 when
the branch was created. That PR is required to compile the retained kernels for
`Ascend950`. Do not compare the candidate against a wheel built from another
CANN version or another #632 revision.

## Run on one Ascend 950

```bash
git fetch origin
git switch experiment/a5-apply-token-bitmask-ab
git pull --ff-only
bash scripts/run_apply_token_bitmask_a5.sh full
```

The runner builds the wheel for `Ascend950`, installs it, runs functional
regression tests, and writes JSON, Markdown, `npu-smi`, commit, and worktree
metadata under `benchmark/results/apply_token_bitmask/<UTC timestamp>/`.

Run a short smoke test first when validating a new environment:

```bash
bash scripts/run_apply_token_bitmask_a5.sh quick
```

## Result acceptance

- All optimized and legacy outputs exactly match the CPU reference.
- No tested case has P50 speedup below `0.95x`.
- Batch-1 vocabularies 128256 and 151936 should show at least `1.50x` P50
  speedup to justify an upstream performance PR.
- Report P50 and P99. Do not replace device-event measurements with host wall
  time.
- Preserve raw per-iteration samples in `results.json`.

## Push measured data

After reviewing `results.md`:

```bash
git add benchmark/results/apply_token_bitmask/<UTC timestamp>
git commit -m "bench: add Ascend 950 apply_token_bitmask results"
git push origin experiment/a5-apply-token-bitmask-ab
```

Do not commit build products, installed wheels, profiler traces, or unrelated
local files.
