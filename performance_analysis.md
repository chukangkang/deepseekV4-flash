# DeepSeek-V4-Flash 输出速度分析

## 当前使用的技术

| 技术 | 配置 |
|---|---|
| **MLA (Multi-head Latent Attention)** | low-rank Q (`q_lora_rank=1024`), grouped low-rank O (`o_lora_rank=1024`, `o_groups=8`) |
| **Sparse Attention + 滑动窗口** | `window_size=128`, 交替 compress_ratio 4/128 |
| **KV 压缩** | `Compressor` (learned gated pooling) + `Indexer` (learned top-k=512 scoring) |
| **TurboQuant KV Cache** | 4-bit 量化 KV 缓存 |
| **FP8 权重** | dense layers 用 FP8 (`ue8m0` scale) |
| **INT4 Expert 权重** | 256 个 routed experts 全用 INT4 |
| **MoE** | 256 experts, 激活 6 个 + 1 shared |
| **Shared Expert 异步** | 在独立 CUDA stream 上并行执行 |
| **Pipeline Parallelism (PP=2)** | 2 卡流水线并行, NCCL send/recv |
| **Hyper-Connections** | `hc_mult=4`, Sinkhorn normalization |
| **Hash Routing** | 前 3 层用 hash-based routing 替代 score routing |
| **Prefill Chunking** | 分块预填充长 prompt |
| **YaRN RoPE Scaling** | 长上下文支持 |

## 为什么还是慢 (bsz=1 仅 ~3 tok/s)

核心原因：**bsz=1 decode 是纯 memory-bandwidth bound**，再多的计算优化也改不了这个本质。

### 1. 模型太大

- **43 层** × (MLA Attention + MoE FFN + 2× HC Sinkhorn) = 每个 token 要读巨量的权重
- 每层激活 **6 个 routed experts + 1 shared expert = 7 个 FFN**，每个 expert 有 w1/w2/w3 三组权重
- 即使 INT4，单层 expert 参数量 = 6 × 3 × 4096 × 2048 × 0.5B ≈ 72MB，43 层 ≈ **3GB** 仅 routed experts
- 加上 attention、shared expert、HC 参数等，每步需要从 HBM 读取的总量巨大

### 2. Pipeline Parallelism 的 bubble

- PP=2 意味着每步都有一次 **NCCL send/recv 往返**（hidden states + next_token）
- bsz=1 时通信/计算比非常差，PP bubble 占比大
- 从第一次请求 745ms→第二次 339ms 可以看到，warmup 后好了一倍，但通信 overhead 仍在

### 3. 每层的额外开销

- **Hyper-Connections**: 每层 2 次 `hc_split_sinkhorn` kernel (attn + ffn)，`hc_mult=4` 意味着 hidden state 是 4 份拷贝
- **KV Compressor + Indexer**: 每层 decode 都要跑压缩逻辑 + Indexer 自身有独立的 Compressor
- **TurboQuant**: 每次 KV read/write 都有 quantize/dequantize 开销

### 4. bsz=1 的根本问题

- GPU 算力利用率极低，大部分时间花在 **读权重** 而不是计算
- 339ms/step ÷ 43 层 ≈ **~8ms/层**，这已经差不多是这个模型大小在 2 卡上的理论下限了

## 可能的提速方向

| 方向 | 预期收益 | 难度 |
|---|---|---|
| **加大 batch size** | 最有效，bsz=8 理论可接近 8× throughput | 低（需有并发请求） |
| **Tensor Parallelism 替代 PP** | 消除 PP bubble，每步少一次往返通信 | 中 |
| **Speculative Decoding** | 用小模型/MTP 草稿，大模型批量验证 | 中高 |
| **CUDA Graph** | 消除 kernel launch overhead | 中 |
| **Expert 权重 offload / prefetch** | 仅预加载被路由到的 expert | 高 |
| **更多卡 (TP=4 或 TP=8)** | 线性减少每卡读取量 | 低（需硬件） |

**最直接的改善是增大 batch size**——当前 bsz=1 时 GPU 几乎只在搬数据，加大 batch 可以分摊权重读取开销，显著提高 throughput（虽然单请求 latency 不变）。

---

## MiniMax M2.7 为什么输出那么快？

MiniMax M2.5 官方声称 **100 tok/s** 输出速度，是其他前沿模型的约 2 倍。M2.7 在此基础上进一步优化。与当前 DeepSeek V4 Flash 的 ~3 tok/s 差距巨大，原因如下：

### 1. 激活参数量差距极大（最核心原因）

| | MiniMax M2 系列 | DeepSeek V4 Flash |
|---|---|---|
| **总参数** | 230B | 更大 |
| **每 token 激活参数** | **~10B** | **远大于 10B** |
| **Expert 数量** | 未公开具体数 | 256 routed + 1 shared |
| **激活 Expert 数** | 未公开 | **6 个 routed + 1 shared = 7 个** |

每步 decode 需要从 HBM 读取的权重量与**激活参数量**成正比。MiniMax M2 仅 10B 激活参数，而 DeepSeek V4 Flash 每 token 激活 7 个 expert × 3 个权重矩阵 × 43 层，加上 attention、HC 等，激活参数量远超 10B。这直接决定了 memory-bandwidth bound 场景下的速度上限。

### 2. 成熟的工业级推理框架

MiniMax M2.7 使用 **vLLM / SGLang** 等成熟框架部署，内置大量优化：

| 优化 | 说明 |
|---|---|
| **CUDA Graph** | 消除 kernel launch overhead，SGLang 配置 `--cuda-graph-max-bs 512` |
| **PagedAttention** | 高效 KV cache 内存管理，支持大 batch |
| **Continuous Batching** | 成熟的请求级调度，GPU 利用率最大化 |
| **FP8 MoE Kernel** | 集成 NVIDIA TensorRT-LLM 的 FP8 MoE kernel，专为 MoE 模型优化 |
| **QK RMS Norm 融合** | 将 QK norm 计算与通信融合为单个 kernel |
| **Flashinfer + TRT-LLM** | `--moe-runner-backend flashinfer_trtllm_routed` 高性能 MoE dispatch |
| **AllReduce 融合** | `--enable-flashinfer-allreduce-fusion` 减少通信开销 |
| **FP8 KV Cache** | `--kv-cache-dtype fp8_e4m3` 减少 KV cache 内存/带宽 |
| **Expert Parallel** | `--enable-expert-parallel` 跨卡分散 expert 计算 |

这些优化在 NVIDIA Blackwell GPU 上可带来 **2.5-2.7× 吞吐提升**（NVIDIA 官方数据）。

### 3. 大 batch 高并发服务

MiniMax 是**云端 API 服务**，同时处理成百上千请求：
- SGLang 配置 `--max-running-requests 512`
- 大 batch 下权重读取被所有请求分摊，GPU 算力利用率极高
- bsz=512 理论上 throughput 可比 bsz=1 高几十到上百倍

**你的场景是 bsz=1 本地推理，MiniMax 是 bsz=几百的云端服务，两者不可直接比较。**

### 4. 更简洁的架构

MiniMax M2 系列（从 M2 开始）放弃了前代 MiniMax-01 的 lightning attention（线性注意力），改用**标准 full softmax attention**，原因是：
- 线性注意力在 precision 上仍有缺陷，影响质量
- 标准 attention 有更成熟的推理优化生态（FlashAttention、PagedAttention 等）
- 基础设施优化足够好时，full attention 速度可以接受

而 DeepSeek V4 Flash 使用了大量非标准组件：
- **Hyper-Connections (HC)** — 非标准残差，4× hidden state 拷贝 + Sinkhorn kernel
- **Compressor + Indexer** — 自定义 KV 压缩/检索机制
- **TurboQuant** — 自定义 4-bit KV cache 量化
- **自定义 sparse_attn Tilelang kernel** — 非 FlashAttention 生态

这些自定义组件无法利用 vLLM/SGLang/TRT-LLM 的成熟优化，必须手写所有 kernel，难以达到同等优化水平。

### 5. 硬件差距

MiniMax API 后端使用 **NVIDIA Blackwell (Ultra) GPU** 集群，多卡 TP/EP 部署：
- `--tensor-parallel-size 4`（至少 4 卡 TP）
- Blackwell 的 HBM 带宽和 FP8 算力远超消费级/上一代 GPU

你的 2 卡 PP=2 配置，带宽和算力都远低于这个水平。

### 总结：速度差距来源

| 因素 | 贡献 | 说明 |
|---|---|---|
| **bsz=1 vs bsz=几百** | **~50-100×** | 最大差距来源，云端高并发 vs 本地单请求 |
| **激活参数量 10B vs 远超 10B** | **~3-5×** | 每步读取权重量的直接差距 |
| **成熟推理框架 vs 手写推理** | **~2-3×** | CUDA Graph、融合 kernel、PagedAttention 等 |
| **Blackwell vs 你的 GPU** | **~2-3×** | HBM 带宽 + FP8 算力差距 |
| **TP=4 vs PP=2** | **~1.5-2×** | TP 无 bubble，PP 有 bubble |

**综合起来：MiniMax 的 100 tok/s 是高并发云端服务的 per-user throughput（总 throughput 除以并发用户数），而你的 3 tok/s 是 bsz=1 单请求延迟受限的实际速度。两者衡量的东西完全不同。**

---

## INT4 量化是否反优化？（4090 场景分析）

### 结论：对 bsz=1 decode，INT4 expert 不是反优化，反而是正优化

有一种说法：「4090 Tensor Core 原生支持 FP8/INT8，但 INT4 没有硬件加速，会走软件模拟」。这个说法**对本代码不成立**。

### 实际 INT4 GEMM 实现

查看 `kernel.py` 的 `int4_gemm_kernel`，实际计算流程：

```
INT4 权重 (packed uint8, HBM) 
  → 解包为 signed int4 (shared memory)
  → cast 为 FP8 (shared memory)
  → FP8 × FP8 T.gemm (Tensor Core)  ← 4090 原生支持
  → 乘以 act_scale × weight_scale
```

**关键：实际的矩阵乘用的是 FP8 Tensor Core，不是 INT4 软件模拟。** INT4 只是存储格式。

### INT4 vs FP8 GEMM 对比

| | INT4 GEMM kernel | FP8 GEMM kernel |
|---|---|---|
| **权重 HBM 读取量** | **K/2 bytes** (INT4 packed) | K bytes (FP8) |
| **Tensor Core 类型** | FP8 × FP8 ✅ | FP8 × FP8 ✅ |
| **block_K** | 32 (= weight group size) | 128 |
| **Pipeline stages** | 2 | 4 |
| **Scale 计算** | 双 scale (act per-128 × weight per-32) | 单 combined scale |
| **额外操作** | uint8 解包 + cast to FP8 | 无 |

### bsz=1 decode 场景分析

bsz=1 时 GEMM shape 为 M=1, N=expert_dim, K=4096，**纯 memory-bandwidth bound**：

- ✅ **INT4 权重只有 FP8 的一半大小** → 从 HBM 加载快约 2×，这是最重要的优势
- ✅ 解包操作在 shared memory 级别完成，延迟远小于 HBM 传输
- ⚠️ block_K=32 导致循环次数是 FP8 的 4 倍，每次迭代有 scale 校正开销
- ⚠️ Pipeline stages 少 (2 vs 4)，延迟隐藏能力弱

**净效果：bsz=1 时，HBM 带宽节省 > 额外计算开销，INT4 对 decode 速度有正面影响。**

### 大 batch 场景（补充说明）

如果 bsz 很大（计算密集），INT4 的劣势会显现：
- block_K=32 使 Tensor Core 利用率低于 block_K=128
- 更多循环迭代 + 解包开销在大矩阵上被放大
- 此时直接 FP8×FP8 GEMM 效率更高

但你当前是 bsz=1，这不是你的瓶颈。

### TurboQuant (4-bit KV Cache) 的真实开销

TurboQuant 的 quantize/dequantize 是 **Python 级别逐元素操作**（`model.py` 的 `TurboQuantKVCache`），不涉及 Tensor Core：
- `_quantize_4bit`: 找 min/max → 量化 → pack 为 uint8
- `_dequantize_4bit`: unpack → 反量化

这部分有真实开销，但 KV cache 元素数远小于 expert 权重，不是主要瓶颈。如要优化，可以用 Triton/TileLang kernel 替代 PyTorch 逐元素操作。
