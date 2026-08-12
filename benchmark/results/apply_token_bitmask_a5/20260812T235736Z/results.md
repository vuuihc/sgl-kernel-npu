# apply_token_bitmask A/B results

- Commit: `558b166a1b8d681ca7d6c852ba1b89facd5c292e`
- Device: `Ascend950PR_9579`
- SOC: `260`
- Timing: `torch.npu.Event(enable_timing=True)`
- Iterations: `100`

| Case | Legacy P50 (ms) | Optimized P50 (ms) | Speedup | P99 speedup |
|---|---:|---:|---:|---:|
| b1_v32000_fp16_sparse | 0.237408 | 0.012190 | 19.476x | 18.771x |
| b1_v128256_fp16_sparse | 0.947396 | 0.025066 | 37.796x | 37.185x |
| b1_v151936_bf16_sparse | 1.156653 | 0.035527 | 32.557x | 31.157x |
| b8_v151936_bf16_sparse | 1.166488 | 0.182100 | 6.406x | 6.376x |
| b1_v200000_bf16_sparse | 1.520744 | 0.041408 | 36.726x | 34.934x |
| b1_v151936_bf16_random | 1.356192 | 0.039714 | 34.149x | 33.595x |
| b1_v151936_bf16_unmasked | 0.083114 | 0.016077 | 5.170x | 3.111x |
| b2_v151936_bf16_sparse | 1.160851 | 0.061702 | 18.814x | 17.832x |
| b32_v151936_bf16_sparse | 1.179318 | 0.868482 | 1.358x | 1.359x |
| b1_v32100_fp16_sparse | 0.244726 | 0.017299 | 14.147x | 13.577x |
| idx8_b16_v151936_bf16 | 1.220371 | 0.235937 | 5.172x | 5.113x |

Geomean P50 speedup: **13.044x**
