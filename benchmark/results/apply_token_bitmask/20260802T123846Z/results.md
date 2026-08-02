# apply_token_bitmask A/B results

- Commit: `9669d67c41febf8a6b5d8b6e3b620236e7e19836`
- Device: `Ascend950PR_9579`
- SOC: `260`
- Timing: `torch.npu.Event(enable_timing=True)`
- Iterations: `100`

| Case | Legacy P50 (ms) | Optimized P50 (ms) | Speedup | P99 speedup |
|---|---:|---:|---:|---:|
| b1_v32000_fp16_sparse | 0.237255 | 0.011331 | 20.939x | 20.347x |
| b1_v128256_fp16_sparse | 0.947242 | 0.023991 | 39.483x | 38.869x |
| b1_v151936_bf16_sparse | 1.155622 | 0.033623 | 34.369x | 33.349x |
| b8_v151936_bf16_sparse | 1.164263 | 0.179123 | 6.500x | 6.480x |
| b1_v200000_bf16_sparse | 1.519372 | 0.039258 | 38.703x | 37.461x |
| b1_v151936_bf16_random | 1.355161 | 0.037698 | 35.948x | 35.474x |
| b1_v151936_bf16_unmasked | 0.109070 | 0.014614 | 7.463x | 1.732x |
| b2_v151936_bf16_sparse | 1.159066 | 0.058238 | 19.902x | 19.507x |
| b32_v151936_bf16_sparse | 1.176770 | 0.867368 | 1.357x | 1.358x |
| b1_v32100_fp16_sparse | 0.244323 | 0.016264 | 15.022x | 14.339x |
| idx8_b16_v151936_bf16 | 1.216758 | 0.231574 | 5.254x | 5.241x |

Geomean P50 speedup: **14.012x**
