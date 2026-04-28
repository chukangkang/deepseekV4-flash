#!/bin/bash
# Launch DeepSeek-V4 PP=2 x TP=8 inference - Node 1 (Pipeline Stage 1)
# Usage: bash scripts/run_pp_node1.sh <ckpt_path> <master_addr> [master_port]
#
# Node 0 (ranks 0-7):  embed + layers 0-21   (Pipeline Stage 0, TP=8)
# Node 1 (ranks 8-15): layers 22-42 + head   (Pipeline Stage 1, TP=8)

set -e

# NCCL networking config (change br0 to your actual interface name)
export NCCL_SOCKET_IFNAME=br0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=br0

CKPT_PATH=${1:?"Usage: $0 <ckpt_path> <master_addr> [master_port]"}
MASTER_ADDR=${2:?"Usage: $0 <ckpt_path> <master_addr> [master_port]"}
MASTER_PORT=${3:-29500}
CONFIG=${4:-"config.json"}

NNODES=2
NPROC_PER_NODE=8
NODE_RANK=1

echo "=== DeepSeek-V4 PP=2 x TP=8 Inference - Node 1 (Stage 1: layers 22-42 + head) ==="
echo "  Master addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Config:      ${CONFIG}"
echo "  GPUs:        ${NPROC_PER_NODE} (global ranks 8-15)"
echo ""

torchrun \
    --nnodes ${NNODES} \
    --nproc-per-node ${NPROC_PER_NODE} \
    --node-rank ${NODE_RANK} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    generate_pp.py \
    --ckpt-path ${CKPT_PATH} \
    --config ${CONFIG} \
    --interactive \
    --max-new-tokens 300 \
    --temperature 0.6
