# Production validation

The production implementation from PR #645 was applied to the latest PR #632
validation baseline and tested separately from the experiment-only legacy op.

## Environment

- Device: `Ascend950PR_9579`
- Compile target: `Ascend950PR_9599`
- CANN: `9.0.0`
- PyTorch: `2.7.1+cpu`
- torch_npu: `2.7.1.post4`

The container-provided `LD_LIBRARY_PATH` placed CANN `devlib` stubs before the
runtime libraries. Validation filtered the `/devlib` entry before running
builds and tests.

## Results

- Ascend 950PR production build: passed
- Boundary correctness: `6/6`
- LLM correctness: `24/24`
- General correctness: `21/21`
- `Ascend910_9382` (A3) compile validation: passed
- `Ascend910B1` (A2) compile validation: passed

All stages completed with exit code 0.

The performance artifacts in this directory come from the same-wheel A/B
branch. The production validation used no public legacy op.
