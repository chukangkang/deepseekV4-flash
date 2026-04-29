# Inference code for DeepSeek models

## Features

- **INT4 expert weights** — Symmetric 4-bit quantization (uint8-packed) for MoE expert weights. Compatible with RTX 4090 / SM89 GPUs that lack native FP4 support.
- **TurboQuant KV cache compression** — Training-free per-token asymmetric quantization of the KV cache to 4-bit (default) or 3-bit, reducing KV memory by ~4× or ~5.3×.
- **PP=2 × TP=8 distributed inference** — Pipeline + Tensor parallelism across 2 nodes × 8 GPUs (~10–12 GB VRAM per GPU).
- **OpenAI-compatible API** — `/v1/chat/completions` endpoint with batched scheduling.

## Quick start

### Step 1: Convert checkpoint

```bash
# INT4 expert weights (recommended for RTX 4090)
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts 256 --model-parallel 8 --expert-dtype int4
```

Or use the convenience script:
```bash
bash scripts/convert_pp2_tp8.sh /path/to/hf-checkpoint /path/to/converted-checkpoint
```

Other `--expert-dtype` options:
| Value | Description |
|-------|-------------|
| `int4` | Symmetric INT4, packed uint8 + FP32 scale (default, **4090-compatible**) |
| `fp8` | FP8 (float8_e4m3fn) + FP8 scale |
| `fp4` | Native FP4 (float4_e2m1fn) — requires SM100+ |

### Step 2: Configure `config.json`

Key fields:
```json
{
  "expert_dtype": "int4",
  "turbo_quant": true,
  "turbo_quant_bits": 4
}
```

- Set `"expert_dtype"` to match the `--expert-dtype` used during conversion.
- Set `"turbo_quant": false` to disable KV cache compression.
- Set `"turbo_quant_bits": 3` for more aggressive 3-bit KV compression.

### Step 3: Run inference

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config config.json --interactive
```

Batch inference:
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config config.json --input-file ${FILE}
```

Multi-node:
```bash
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path ${SAVE_PATH} --config config.json --input-file ${FILE}
```

## PP=2 x TP=8 on 2 Nodes x 8x RTX 4090 (24GB) — Recommended

Pipeline Parallelism (PP=2) + Tensor Parallelism (TP=8) across 2 machines, each with 8x RTX 4090 24GB GPUs.
- **Node 0** (PP stage 0): embedding + layers 0–21 (22 layers)
- **Node 1** (PP stage 1): layers 22–42 (21 layers) + head + MTP

Mixed precision: FP8 non-expert weights + **INT4 expert weights** + TurboQuant 4-bit KV cache.

**Memory per GPU**: ~10–12 GB (each node holds ~half the model layers)

### Prerequisites
- Both nodes: PyTorch >= 2.10, CUDA, NCCL
- Nodes must be network-reachable (NCCL uses TCP for inter-node communication)
- Converted checkpoint accessible on both nodes at the same path

### Step 1: Convert checkpoint (run once, on any machine)
```bash
bash scripts/convert_pp2_tp8.sh /path/to/hf-checkpoint /path/to/pp2-tp8-checkpoint
```
This runs `convert.py --expert-dtype int4` and generates 8 files: `model{0..7}-mp8.safetensors`.
Copy them + tokenizer files to both nodes.

### Step 2: Launch inference

**Node 0 (master — PP stage 0):**
```bash
bash scripts/run_pp_node0.sh /path/to/pp2-tp8-checkpoint [master_ip] [port]
```

**Node 1 (worker — PP stage 1):** (replace `MASTER_IP` with Node 0's IP)
```bash
bash scripts/run_pp_node1.sh /path/to/pp2-tp8-checkpoint MASTER_IP [port]
```

### How it works
1. `torchrun` launches 8 processes per node (16 total). Global ranks 0–7 = Node 0, 8–15 = Node 1.
2. Each node creates a TP sub-group of 8 GPUs for intra-node all-reduce/all-gather.
3. Matching TP ranks across nodes communicate via point-to-point send/recv (rank i ↔ rank i+8).
4. Stage 0 runs embedding + first 22 transformer blocks, sends hidden states to stage 1.
5. Stage 1 receives hidden states, runs remaining 21 blocks + head, sends logits back.

### Environment tuning
Edit the `NCCL_SOCKET_IFNAME` in the launch scripts to match your network interface:
```bash
export NCCL_SOCKET_IFNAME=br0     # Change to your interface (eth0, ens3, etc.)
export NCCL_IB_DISABLE=1          # Set to 0 if InfiniBand is available
export NCCL_SOCKET_FAMILY=AF_INET # Force IPv4
export NCCL_DEBUG=INFO             # Debug NCCL connectivity
```

### TurboQuant KV Cache Compression

TurboQuant reduces KV cache memory via per-token low-bit asymmetric quantization. This is a **runtime-only** optimization — no checkpoint re-conversion needed.

**How it works:**
1. When writing to the KV cache, each token's KV vector is quantized to N-bit with per-token scale + zero-point
2. When reading for attention, the cached values are dequantized back to BF16
3. During prefill, raw BF16 tensors are used directly (no cache read)

**Configuration in `config.json`:**

| Field | Default | Description |
|-------|---------|-------------|
| `turbo_quant` | `true` | Enable/disable KV cache compression |
| `turbo_quant_bits` | `4` | Quantization bits: `4` (~4× compression) or `3` (~5.3× compression) |

**Memory savings (per token per layer):**
| Mode | KV size | vs BF16 |
|------|---------|---------|
| BF16 (off) | 1024 bytes | 1× |
| 4-bit | ~260 bytes | ~4× |
| 3-bit | ~196 bytes | ~5.3× |

To disable: set `"turbo_quant": false` in `config.json`. No other changes needed.

### OpenAI-compatible API service

This repository also provides an OpenAI-compatible HTTP server via `openai_server.py`.

Available endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Only global rank 0 serves HTTP traffic. Other ranks participate in distributed inference only.

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### Launch on 2 nodes x 8 GPUs

**Node 0 (master):**
```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank 0 --master-addr MASTER_IP --master-port 29500 openai_server.py --ckpt-path /path/to/pp2-tp8-checkpoint --config config.json --host 0.0.0.0 --port 8000 --model-name deepseek-v4-flash
```

**Node 1 (worker):**
```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank 1 --master-addr MASTER_IP --master-port 29500 openai_server.py --ckpt-path /path/to/pp2-tp8-checkpoint --config config.json --host 0.0.0.0 --port 8000 --model-name deepseek-v4-flash
```

#### Launch on a single 16-GPU node
```bash
torchrun --nproc-per-node 16 openai_server.py --ckpt-path /path/to/pp2-tp8-checkpoint --config config.json --host 0.0.0.0 --port 8000 --model-name deepseek-v4-flash
```

#### Recommended flags for long-context experiments
```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank 0 --master-addr MASTER_IP --master-port 29500 openai_server.py --ckpt-path /path/to/pp2-tp8-checkpoint --config config.json --host 0.0.0.0 --port 8000 --model-name deepseek-v4-flash --max-seq-len 131072 --max-batch-size 8 --max-batch-total-tokens 65536 --prefill-chunk-size 256 --release-kv-after-batch
```

Useful flags:
- `--max-seq-len`: override runtime context length for long-context experiments
- `--max-batch-size`: max requests per scheduling batch
- `--max-batch-total-tokens`: token budget per sub-batch to avoid oversized mixed batches
- `--prefill-chunk-size`: chunk size used during prompt prefill
- `--release-kv-after-batch`: release KV storage after each batch for a safer memory profile

#### Test the API

Windows PowerShell:
```bash
curl http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}],\"max_tokens\":128,\"temperature\":0.6}"
```

Linux/macOS:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好，介绍一下你自己"}],"max_tokens":128,"temperature":0.6}'
```

For OpenAI-compatible clients, use:
- Base URL: `http://HOST:8000/v1`
- Model: `deepseek-v4-flash`

#### Request example

Minimal JSON payload for `POST /v1/chat/completions`:

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {
      "role": "user",
      "content": "Write a short introduction about DeepSeek-V4-Flash."
    }
  ],
  "max_tokens": 128,
  "temperature": 0.6,
  "top_p": 0.95
}
```

Common request fields:
- `model`: must match the name passed by `--model-name`
- `messages`: OpenAI-format chat messages
- `max_tokens`: maximum generated tokens
- `temperature`: set to `0` for greedy decoding
- `top_p`: nucleus sampling threshold
- `stop`: optional stop string or string list

#### Long-context tuning suggestions

Suggested rollout order:
- `32k`
- `64k`
- `128k`
- `256k`
- `512k`
- `1M`

Practical tuning hints:
- Start with `--release-kv-after-batch` enabled for stability during early tests.
- Keep `--prefill-chunk-size` conservative, usually `256` or `512`.
- Set `--max-batch-total-tokens` to avoid one large request monopolizing a batch.
- Increase `--max-seq-len` gradually and watch both GPU memory and inter-node communication time.

#### Troubleshooting

- If only one node returns immediately and the other appears stuck, first verify `MASTER_IP`, `--master-port`, and NCCL network interface settings.
- If the service starts but no HTTP port is exposed, confirm you are checking the machine hosting global rank 0.
- If you hit context length errors, increase `--max-seq-len` explicitly at startup.
- If mixed long and short prompts cause unstable latency, lower `--max-batch-size` or set `--max-batch-total-tokens`.
- If memory grows too aggressively during long-context tests, enable `--release-kv-after-batch` and reduce batch token budget.

#### Notes
- Current distributed path is designed for `PP=2 x TP=8`, which expects 16 GPUs total.
- For long-context rollout, increase gradually: `32k -> 64k -> 128k -> 256k -> 512k -> 1M`.
- The current implementation already reduces decode communication by sampling on stage 1 and only returning the next token.

---

## (Legacy) TP=16 on 2 Nodes — Not Recommended

> **Note**: Pure TP=16 across 2 nodes fails because `o_groups=8 < world_size=16`, causing
> `n_local_groups = 8 // 16 = 0` and a shape error. Use the PP=2 x TP=8 mode above instead.

<details>
<summary>Old instructions (for reference only)</summary>

### Step 1: Convert checkpoint
```bash
bash scripts/convert_fp8_16gpu.sh /path/to/hf-checkpoint /path/to/fp8-checkpoint
```

### Step 2: Launch
```bash
# Node 0
bash scripts/run_node0.sh /path/to/fp8-checkpoint [master_ip] [port]
# Node 1
bash scripts/run_node1.sh /path/to/fp8-checkpoint MASTER_IP [port]
```
</details>
