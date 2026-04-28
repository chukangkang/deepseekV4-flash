#!/bin/bash
# Launch DeepSeek-V4 FP8 inference on Node 0 (master node)
# Usage: bash scripts/run_node0.sh <ckpt_path> [master_addr] [master_port]
#
# Node 0 hosts GPU ranks 0-7, Node 1 hosts GPU ranks 8-15
# Total: 2 nodes x 8 GPUs = 16-way model parallelism

set -e

# Force NCCL to use the correct network interface (change eth0 to your actual interface name)
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=eth0

CKPT_PATH=${1:?"Usage: $0 <ckpt_path> [master_addr] [master_port]"}
MASTER_ADDR=${2:-"$(hostname -I | awk '{print $1}')"}
MASTER_PORT=${3:-29500}
CONFIG=${4:-"config.json"}

NNODES=2
NPROC_PER_NODE=8
NODE_RANK=0

echo "=== DeepSeek-V4 FP8 Inference - Node 0 (Master) ==="
echo "  Master addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Config:      ${CONFIG}"
echo "  GPUs:        ${NPROC_PER_NODE} (ranks 0-7)"
echo ""
echo "Make sure Node 1 is launched with the same MASTER_ADDR and MASTER_PORT."
echo ""

torchrun \
    --nnodes ${NNODES} \
    --nproc-per-node ${NPROC_PER_NODE} \
    --node-rank ${NODE_RANK} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    generate.py \
    --ckpt-path ${CKPT_PATH} \
    --config ${CONFIG} \
    --interactive \
    --max-new-tokens 300 \
    --temperature 0.6
