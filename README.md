# Inference code for DeepSeek models

First convert huggingface model weight files to the format of this project.
```bash
export EXPERTS=256
export MP=4
export CONFIG=config.json
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Then chat with DeepSeek model at will!
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

Or batch inference from file.
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

Or multi nodes inference.
```bash
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

If you want to use fp8, just remove `"expert_dtype": "fp4"` in `config.json` and specify `--expert-dtype fp8` in `convert.py`.

## PP=2 x TP=8 on 2 Nodes x 8x RTX 4090 (24GB) — Recommended

Pipeline Parallelism (PP=2) + Tensor Parallelism (TP=8) across 2 machines, each with 8x RTX 4090 24GB GPUs.
- **Node 0** (PP stage 0): embedding + layers 0–21 (22 layers)
- **Node 1** (PP stage 1): layers 22–42 (21 layers) + head + MTP

Mixed precision: FP8 non-expert weights + FP4 expert weights.

**Memory per GPU**: ~10–12 GB (each node holds ~half the model layers)

### Prerequisites
- Both nodes: PyTorch >= 2.10, CUDA, NCCL
- Nodes must be network-reachable (NCCL uses TCP for inter-node communication)
- Converted checkpoint accessible on both nodes at the same path

### Step 1: Convert checkpoint (run once, on any machine)
```bash
bash scripts/convert_pp2_tp8.sh /path/to/hf-checkpoint /path/to/pp2-tp8-checkpoint
```
This generates 8 files: `model{0..7}-mp8.safetensors`. Copy them + tokenizer files to both nodes.

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
