# 单机单卡 Ascend 950 性能优化贡献路线

本文整理仅使用一台单机单卡 Ascend 950 时，适合在
`sgl-kernel-npu` 和 SGLang NPU 后端推进的高价值 PR 方向。

目标不是罗列所有可能的优化，而是筛选出满足以下条件的工作：

1. 单卡可以独立复现、开发和验收。
2. 优化对象位于 SGLang 当前仍在使用的真实路径上。
3. 可以建立可信的 kernel microbenchmark。
4. 最好还能通过单卡可运行模型证明端到端收益。
5. PR 边界清楚，不依赖多机通信、超大模型或尚不可获得的 CI。
6. 正确性和性能结论均能由测试数据直接支持。

## 1. 结论与优先级

建议优先级如下：

| 优先级 | 方向 | 仓库 | 性能确定性 | 潜在价值 | 工程风险 |
|---|---|---|---:|---:|---:|
| P0 | `apply_token_bitmask` 二维并行 | `sgl-kernel-npu` | 很高 | 高 | 低 |
| P0 | MLA V up-projection 迁移官方算子 | SGLang | 高 | 很高 | 低到中 |
| P1 | HiCache KV 传输合并与去同步 | 两个仓库 | 中高 | 很高 | 中 |
| P1 | speculative `build_tree` 固定开销优化 | 两个仓库 | 高 | 中 | 低 |
| P2 | A5 LoRA decode kernel | `sgl-kernel-npu` | 中 | 高 | 高 |
| 基础设施 | NPU capability 检测与 A5 feature gate | SGLang | 不适用 | 高 | 低 |

如果目标是尽快形成第一个具有可信性能数字的 PR，推荐顺序是：

```text
apply_token_bitmask
    -> MLA 官方算子迁移
    -> HiCache transfer
    -> build_tree
    -> LoRA A5 kernel
```

## 2. 选择标准

### 2.1 什么叫“确定性”

一个方向至少需要满足以下三点：

- **瓶颈有代码证据**：例如 batch=1 只启动一个 AIV，而不是仅凭 profiler
  猜测。
- **收益可被隔离测量**：baseline 和 candidate 使用相同输入、相同软件版本、
  相同计时方法。
- **正确性可穷举关键边界**：包含 dtype、shape、非对齐尺寸、可选参数和极端
  mask 等场景。

### 2.2 什么叫“高价值”

优先选择满足下列条件之一的路径：

- decode 每 token 都会执行。
- 位于 structured output、speculative decoding、HiCache 等关键功能上。
- 当前实现阻碍 A5 正常构建或迫使主仓保留后端特例。
- 优化不仅改善 microbenchmark，还能改善 TPOT、TTFT 或 prefix-hit latency。

### 2.3 不应使用的证明方式

- 不能用 host wall-clock 代替 kernel device 时间。
- 不能只给平均值，不给 P50/P99 和原始样本。
- 不能只测规则小 shape，忽略真实模型 shape。
- 不能用不同 CANN、不同 `torch_npu` 或不同构建选项的 wheel 做 A/B。
- 不能把 correctness 修复、构建修复和性能重写混成一个难以归因的大 PR。

## 3. 统一测试与计时规范

### 3.1 环境信息

每次 benchmark 至少记录：

- commit SHA。
- Ascend SOC 名称。
- `npu-smi info`。
- CANN、PyTorch、`torch_npu` 和 `sgl-kernel-npu` 版本。
- dtype、shape、warmup 次数和测量次数。
- 是否启用 indices、padding、graph mode 或其他特殊路径。

### 3.2 Device 时间

使用 NPU timing event：

```python
start = torch.npu.Event(enable_timing=True)
end = torch.npu.Event(enable_timing=True)

start.record()
op()
end.record()
torch.npu.synchronize()

latency_ms = start.elapsed_time(end)
```

推荐：

- warmup 不少于 20 次。
- 正式测量不少于 100 次。
- 输出 mean、P50、P90、P99、min、max、stdev。
- 保存每轮原始 latency 样本，便于排查抖动和离群点。
- baseline 和 candidate 交替执行，降低温度、频率和缓存漂移。

Host wall-clock 可以用于完整请求 E2E，但不能用于宣称 kernel latency。

### 3.3 正确性

kernel PR 至少覆盖：

- 所有声明支持的 dtype。
- batch=1 和多 batch。
- 最小尺寸、典型尺寸、非对齐尺寸和大尺寸。
- 关键可选参数开启和关闭。
- baseline、candidate 和独立 reference 三方对比。
- 对 in-place operator 验证未选中区域没有被修改。

浮点算子必须说明误差阈值来源；bitmask、index、copy 类算子应优先要求精确一致。

### 3.4 性能验收

通用建议：

- 核心目标 shape 的 P50 至少提升 `1.20x`。
- 如果 PR 引入较多复杂度，核心 shape 应达到 `1.50x` 以上。
- 任一重要 shape 不应低于 baseline 的 `0.95x`。
- 同时检查 P99，避免平均性能提升但尾延迟恶化。
- E2E 收益应与 kernel 调用频次和 kernel latency 变化基本一致。

## 4. P0：`apply_token_bitmask` 二维并行

### 4.1 业务路径

该算子用于 structured output / JSON Schema 等 constrained decoding 路径，
对 logits 应用压缩 bitmask：

- bit 为 0：对应 logit 写为 `-inf`。
- bit 为 1：保留原 logit。
- `indices` 可限制只处理部分 batch row。

structured output decode 经常是 batch=1，因此小 batch 的并行策略非常重要。

### 4.2 当前代码证据

涉及文件：

- `csrc/apply_token_bitmask/op_host/apply_token_bitmask.cpp`
- `csrc/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp`
- `tests/python/sgl_kernel_npu/test_apply_token_bitmask.py`

原始 host 侧分核方式为：

```cpp
uint32_t blockDim =
    std::min(static_cast<int64_t>(numIndices), coreNum);
```

这意味着：

- batch=1 时只启动一个 AIV。
- batch=2 时最多只使用两个 AIV。
- vocab 即使达到 128K 或 151K，也不会增加并行度。

kernel 内部还存在逐 `int32`、逐 bit 的标量访问：

```cpp
int32_t packed = bitmaskLocal.GetValue(intIdx);
for (uint32_t bitIdx = 0; bitIdx < 32; bitIdx++) {
    if (((packed >> bitIdx) & 1) == 0) {
        outLocal.SetValue(i, negInf);
    }
}
```

第一阶段不必同时重写标量 bit 处理。仅把工作划分从“按 row”改成
“按 row × vocab tile”，就有明确的低风险并行收益。

### 4.3 第一阶段实现

推荐 flattened work index：

```text
total_work = num_rows * num_tiles
work_idx   = block_idx + k * block_dim
row_id     = work_idx / num_tiles
tile_id    = work_idx % num_tiles
```

tile 长度需要同时满足：

- 是 256 elements 的整数倍，满足 logits 和 bitmask DataCopy 对齐。
- 不超过 UB 可容纳的最大 tile。
- 小 batch 时主动缩小 tile，使 `num_rows * num_tiles` 尽量覆盖所有 AIV。

当前实验分支已经实现：

- `torch.ops.npu.apply_token_bitmask_legacy`：原始 row partition。
- `torch.ops.npu.apply_token_bitmask`：candidate row-by-tile partition。
- 两个入口共享输入检查、padding、indices、host allocation 和 stream 管理。
- benchmark 在同一个 wheel 中交替执行两个入口。

这样 A/B 的差异只剩 tile 大小和 AIV work partition，归因比构建两个不同 wheel
更可靠。

### 4.4 Benchmark 矩阵

最低要求：

| 维度 | 取值 |
|---|---|
| batch | 1、2、8、32 |
| vocab | 32000、128256、151936、200000 |
| dtype | FP16、BF16；FP32 用于兼容性 |
| mask | sparse、random、all-unmasked |
| indices | 关闭；16 rows 中选择 8 rows |
| 对齐 | 256 对齐和非 256 对齐 |

重点 shape：

- `batch=1, vocab=128256, fp16`
- `batch=1, vocab=151936, bf16`
- `batch=1, vocab=200000, bf16`
- `batch=8, vocab=151936, bf16`

现有脚本：

```bash
bash scripts/run_apply_token_bitmask_a5.sh quick
bash scripts/run_apply_token_bitmask_a5.sh full
```

结果会写入：

```text
benchmark/results/apply_token_bitmask/<UTC timestamp>/
```

其中包含：

- `results.json`：原始样本和统计结果。
- `results.md`：可直接贴入 PR 的表格。
- `functional-tests.txt`：正确性测试输出。
- `npu-smi.txt`：硬件环境。
- `commit.txt` 和 `git-status.txt`：代码状态。

### 4.5 E2E 验证

推荐模型：

- Qwen3-8B。

推荐场景：

- JSON Schema constrained decoding。
- 单并发和 16 并发分别测试。
- 固定 prompt、schema、输出 token 数和随机种子。

指标：

- TPOT。
- TTFT。
- output tokens/s。
- JSON parse success。
- schema valid rate。

### 4.6 PR 验收

建议正式 PR 的最低门槛：

- optimized 与 legacy、CPU reference 精确一致。
- batch=1 的 128256 和 151936 vocab P50 至少 `1.50x`。
- 重要 shape 没有低于 `0.95x` 的回退。
- P99 没有明显恶化。
- 正式 PR 删除实验用 legacy public schema，只保留必要测试和 candidate 实现。
- 如果第一阶段收益不足，再评估 AscendC vector API 重写逐 bit 标量循环。

## 5. P0：MLA V up-projection 迁移官方算子

这是跨仓方向，主要修改 SGLang，但会减少对本仓旧 kernel 的依赖。

### 5.1 当前路径

SGLang 文件：

```text
python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
```

当前流程大致为：

```python
attn_bmm_output = torch.empty(...)
attn_output = attn_output.contiguous()
torch.ops.npu.batch_matmul_transpose(
    attn_output,
    m.w_vc,
    attn_bmm_output,
)
```

本仓自定义 `batch_matmul_transpose` 存在几个问题：

- README 和实现主要面向 A2/A3。
- kernel 使用旧架构约束。
- [Issue #461](https://github.com/sgl-project/sgl-kernel-npu/issues/461)
  已记录 shape 正确性失败。
- A5 构建 PR
  [#632](https://github.com/sgl-project/sgl-kernel-npu/pull/632)
  正在从 A5 构建中移除该 kernel。

### 5.2 候选实现

使用官方算子：

```python
attn_output = torch_npu.npu_transpose_batchmatmul(
    attn_output,
    m.w_vc,
    perm_y=(1, 0, 2),
)
```

潜在收益：

- 去掉显式 output allocation。
- 可能去掉额外 `.contiguous()`。
- 使用官方维护的 A5 kernel。
- 减少 SGLang 对本仓 A2/A3-only kernel 的依赖。
- 简化后续 A5 构建矩阵。

vLLM Ascend 的 MLA 路径已有类似用法，可作为 API 和 shape 处理参考，但仍需要
以 SGLang 的真实 layout 做独立正确性验证。

### 5.3 测试

microbenchmark：

- 从 DeepSeek-V2-Lite 实际运行中收集全部调用 shape。
- tokens 取 1、2、8、32。
- FP16/BF16。
- contiguous 和真实上游 layout。
- 比较自定义 op、官方 op 和 PyTorch reference。

E2E：

- DeepSeek-V2-Lite-Chat 16B。
- 单卡 decode。
- 测量 MLA 子阶段 latency、TPOT 和 tokens/s。

### 5.4 PR 定位

该 PR 应描述为：

> Replace the custom A2/A3-only MLA V up-projection kernel with the official
> Ascend operator, restoring the A5 path and reducing allocation/copy overhead.

它兼具 correctness、A5 enablement 和性能价值，但不能在没有数据时只宣称性能优化。

## 6. P1：HiCache `transfer_kv_dim_exchange`

### 6.1 当前代码证据

本仓文件：

```text
csrc/transfer_kv_dim_exchange/op_host/transfer_kv_dim_exchange.cpp
python/sgl_kernel_npu/sgl_kernel_npu/kvcacheio.py
```

SGLang 调用方：

```text
python/sglang/srt/mem_cache/pool_host/mha.py
python/sglang/srt/mem_cache/pool_host/mla.py
```

当前 host 路径包含：

```cpp
auto device_indices_cpu = device_indices.cpu();
auto host_indices_cpu = host_indices.cpu();

for (const auto i : c10::irange(num_pages)) {
    aclrtMemcpy2dAsync(...);  // K
    aclrtMemcpy2dAsync(...);  // V
}
```

主要成本：

1. 每次传输都把 indices 同步到 CPU。
2. host 按 page 循环。
3. 每页 K/V 分别发起 DMA。
4. MLA 的 index-K 通过第二次完整 op 调用传输。
5. 第二次调用会重复 indices D2H 和逐页循环。

### 6.2 推荐拆分

不要一次改完所有内容，建议拆成两个 PR。

#### PR A：消除重复 metadata 工作

- 一次调用同时处理 K、V 和可选 index-K。
- indices 只准备一次。
- MLA 不再重复调用完整 transfer op。
- 保持每页 DMA 语义不变，降低 correctness 风险。

#### PR B：连续页 coalescing

- 检测 device page 和 host page 是否形成连续区间。
- 把多个相邻 page 合并为一次大 DMA。
- random page 继续走原始逐页路径。
- 合并逻辑必须同时验证 H2D 和 D2H。

### 6.3 Microbenchmark

维度：

| 维度 | 取值 |
|---|---|
| cache | MHA、MLA |
| direction | H2D、D2H |
| pages | 1、4、16、64、256 |
| page layout | 全连续、部分连续、随机 |
| payload | K、K+V、K+V+index-K |
| page size | 使用 SGLang 真实配置中的 page size |

指标：

- Device transfer duration。
- Host enqueue duration。
- 有效 payload bytes / device duration。
- DMA launch 数。
- P50/P99。

注意：只有明确统计真实 payload bytes，才能报告有效带宽；不要把 metadata 或
padding 当成有效 payload。

### 6.4 E2E

单卡即可构造：

- CPU host cache + 一张 950。
- 固定 prefix。
- cold miss、partial hit、full hit。
- 比较 prefix-hit TTFT 和 cache load-back latency。

这个方向的系统价值很高，但异步生命周期、pinned memory 和 stream ordering
必须优先保证正确，不能只看 memcpy 带宽。

## 7. P1：speculative `build_tree` 固定开销

### 7.1 当前代码证据

本仓：

```text
csrc/build_tree/op_host/build_tree.cpp
csrc/build_tree/op_kernel/build_tree_kernel.cpp
```

SGLang：

```text
python/sglang/srt/speculative/eagle_utils.py
```

当前每轮可能发生：

- 构造 CPU tiling tensor。
- tiling H2D copy。
- 分配 workspace tensor。
- Python 使用 `torch.full(..., -1)` 初始化 retrieve buffer。
- kernel 再次初始化 retrieve buffer。

其中 `workspace_in` 在 kernel 中没有实际读取，是最容易独立验证的删除点。

### 7.2 推荐拆分

#### PR A：删除未使用 workspace

- 从 host launch 和 kernel signature 删除 workspace。
- 证明 kernel 没有读取该地址。
- 比较 launch latency 和 allocation 次数。

#### PR B：缓存或消除固定 tiling H2D

- 按 shape/参数缓存 tiling。
- 或把少量 tiling 参数改为 scalar launch argument。
- 明确 cache 生命周期和多 stream 行为。

#### PR C：retrieve buffer 使用 `empty`

- kernel 已完整覆盖的 buffer 不需要 Python 预填 `-1`。
- 该改动主要位于 SGLang。
- 需要与开放 PR
  [sglang#26932](https://github.com/sgl-project/sglang/pull/26932)
  协调；该 PR 当前主要处理 CUDA 路径，不能重复修改相同代码而不说明 NPU 差异。

### 7.3 Benchmark

建议 shape：

- batch：1、4、16。
- top-k：4、8。
- draft tokens：5、8、16。
- tree depth：覆盖 SGLang 实际 EAGLE 配置。

指标：

- 单次 op device latency。
- host enqueue latency。
- 每轮 allocation 次数。
- EAGLE verify step latency。
- speculative decode TPOT。

这个方向容易形成小 PR，但绝对收益可能小于 bitmask 和 HiCache。

## 8. P2：A5 LoRA decode kernel

### 8.1 当前代码证据

相关 kernel：

```text
csrc/lora/op_kernel/sgemmv_shrink_kernel.cpp
csrc/lora/op_kernel/sgemmv_expand_kernel.cpp
```

SGLang backend：

```text
python/sglang/srt/lora/backend/ascend_backend.py
```

风险信号：

- decode 小 batch 时，一 token 基本映射一个 AIV。
- shrink 逐 rank 扫 hidden dimension。
- A5 的多 vector core 在单 token 场景利用不足。
- expand 对 rank 8/16/32/64 使用手工 reduction。

潜在方向：

- token × rank 或 token × hidden block 二维并行。
- shrink/expand 使用更合理的 AIV work partition。
- 对适合的 shape 评估 CUBE 实现。
- 减少 SGLang 每层 intermediate/output allocation。

### 8.2 为什么不建议作为第一项

历史 CUBE LoRA 实现曾因工程和正确性问题被 revert。重新推进必须比普通 kernel PR
提供更强的：

- rank/shape 覆盖。
- 多 adapter 混合场景。
- inactive adapter 和 padding 场景。
- FP16/BF16 误差分析。
- decode E2E 数据。

### 8.3 Benchmark

至少覆盖：

- batch/token：1、2、8、16、32。
- rank：8、16、32、64。
- hidden size：使用 Qwen/Llama 真实尺寸。
- adapter：单 adapter、多 adapter、部分 token 无 adapter。
- shrink 和 expand 分别测量。

PR 门槛建议高于普通优化：

- 核心 batch=1 shape 至少 `1.50x`。
- 全部 rank 正确。
- E2E LoRA decode 有可见收益。
- 没有引入新的 model-specific 分支。

## 9. 基础设施 PR：NPU capability 与 A5 feature gate

这个方向不是性能优化，但会降低所有 A5 性能工作的验证成本。

### 9.1 正确抽象

应在 SGLang `DeviceMixin` / NPU platform 中实现统一的：

```python
get_device_capability() -> DeviceCapability
```

读取：

```python
torch_npu.npu._backends.get_soc_version()
```

建议映射：

| SOC | capability |
|---|---|
| Ascend910B* | `(2, 0)` |
| Ascend910_93* | `(3, 0)` |
| Ascend950* | `(5, 0)` |

不要使用公开的 `torch_npu.npu.get_device_capability()` 判断真实 NPU 代际；该 API
主要用于 CUDA capability 兼容环境变量。

### 9.2 必须保持的边界

- **capability**：判断某项功能或 kernel 是否可用。
- **memory capacity**：决定 chunked prefill、graph batch size 等资源默认值。

不能因为 950 常见为 96 GB，就用显存容量识别芯片；也不能用芯片代际替换所有容量
分档，因为同代不同 SKU 和本地内存预留仍会改变可用资源。

### 9.3 A5-only gate

MXFP4/MXFP8 的 A5 校验应放在共享 NPU linear method 或共享 helper，而不是只在两个
quant config 中零散增加判断。需要覆盖：

- online MXFP8。
- online MXFP4 W4A8。
- online MXFP4 W4A4。
- ModelSlim offline 对应 scheme。

文档也不能机械地把所有 `A2, A3` 全局替换为 `A2, A3, A5`，否则会把未验证功能
错误声明为已支持。

## 10. 暂时不建议投入的方向

### 10.1 A5 fused MoE

[PR #602](https://github.com/sgl-project/sgl-kernel-npu/pull/602) 已经在推进。
除非 maintainer 明确要求协作，不建议另起重复实现。

### 10.2 `causal_conv1d` PTO

[PR #601](https://github.com/sgl-project/sgl-kernel-npu/pull/601) 已覆盖该方向，
不适合作为新的独立选题。

### 10.3 DeepEP

DeepEP 的核心价值是跨卡通信：

- 单卡无法证明 dispatch/combine 通信收益。
- 单卡 microbenchmark 容易只测到不代表真实部署的固定开销。
- 缺少多卡拓扑时，结论不能支撑高信用性能 PR。

可以修单卡可复现 correctness 问题，但不应把单卡结果包装成 DeepEP 性能优化。

### 10.4 即将从 A5 构建移除的旧 kernel

A5 build PR #632 正在移除一批不适合 A5 的旧实现，例如：

- `batch_matmul_transpose`
- `mla_preprocess`
- `lightning_indexer`
- `tri_inv`
- 部分 GDN / SoftFP8 kernel

除非主仓仍存在明确消费路径并且 maintainer 希望保留，否则不要只为了让旧测试通过而
修这些 kernel。优先迁移到官方算子或当前维护的实现。

## 11. 推荐 PR 拆分

建议保持每个 PR 只有一个主要性能假设。

### PR 1：bitmask work partition

```text
目标：batch=1 使用多个 AIV
包含：kernel partition、正确性测试、microbenchmark
不包含：标量 bit loop 向量化、SGLang 大规模重构
```

### PR 2：SGLang MLA official op

```text
目标：移除 A5 不支持的 custom BMM 路径
包含：官方 op、shape tests、DeepSeek-V2-Lite 数据
不包含：其他 MLA kernel 重写
```

### PR 3：HiCache metadata de-duplication

```text
目标：K/V/index-K 一次处理 metadata
包含：调用接口、正确性、传输 benchmark
不包含：连续页 DMA coalescing
```

### PR 4：HiCache contiguous-page coalescing

```text
目标：减少 DMA launch 数
包含：连续区间识别、fallback、带宽与 TTFT
依赖：PR 3
```

### PR 5：build_tree workspace removal

```text
目标：删除确定未使用的 allocation
包含：signature、host launch、microbenchmark
不包含：retrieve buffer 和 tiling cache
```

这种拆分便于 maintainer：

- 单独验证每个假设。
- 判断收益是否值得复杂度。
- 回滚单个优化。
- 避免跨仓冲突拖住全部工作。

## 12. 数据回传与分析模板

测试机完成后，建议把结果目录提交到实验分支，而不是直接放进正式 PR：

```bash
git add benchmark/results/<operator>/<timestamp>
git commit -m "bench: add Ascend 950 <operator> results"
git push
```

分析时按以下顺序：

1. correctness 是否全部通过。
2. 环境、commit 和构建参数是否一致。
3. raw samples 是否存在明显 warmup、频率或系统噪声。
4. P50 是否证明 steady-state 收益。
5. P99 是否有回退。
6. 收益是否集中在真实业务 shape。
7. E2E 改善是否与 kernel 数据方向一致。
8. 优化复杂度是否值得收益。

正式 PR 中建议只保留：

- 可复现 benchmark 脚本。
- 汇总结果表。
- 必要 profiler 证据。
- 正确性测试。

不要提交：

- wheel、`.so`、build 目录。
- 大型 profiler trace。
- 本地环境缓存。
- 与 PR 无关的调试日志。
- 实验阶段使用、正式接口不需要的 baseline schema。

## 13. 当前实验分支说明

`experiment/a5-apply-token-bitmask-ab` 当前叠加在 A5 build PR #632 的 head 上，
因为该 PR 负责：

- 把 `SOC_VERSION` 正确传给 CMake。
- 从 Ascend 950 构建中排除不支持的旧 kernel。

当前 A/B baseline 固定为：

```text
c3c34bafce9926ed519fbdbafaa694ccdbf9e176
```

bitmask candidate commit 为：

```text
9669d67c41febf8a6b5d8b6e3b620236e7e19836
```

正式提 PR 前应重新检查 #632 状态：

- 如果 #632 已合入：把性能 commit rebase 到最新 `upstream/main`。
- 如果 #632 未合入：明确声明依赖，或者等待其合入后再提交。
- 不要把 #632 的构建改动计入 bitmask 性能收益。
