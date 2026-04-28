#!/bin/bash
# Launch DeepSeek-V4 FP8 batch inference on Node 0 (master node)
# Usage: bash scripts/run_batch_node0.sh <ckpt_path> <input_file> [master_addr] [master_port]

set -e

CKPT_PATH=${1:?"Usage: $0 <ckpt_path> <input_file> [master_addr] [master_port]"}
INPUT_FILE=${2:?"Usage: $0 <ckpt_path> <input_file> [master_addr] [master_port]"}
MASTER_ADDR=${3:-"$(hostname -I | awk '{print $1}')"}
MASTER_PORT=${4:-29500}
CONFIG=${5:-"config.json"}

torchrun \
    --nnodes 2 \
    --nproc-per-node 8 \
    --node-rank 0 \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    generate.py \
    --ckpt-path ${CKPT_PATH} \
    --config ${CONFIG} \
    --input-file ${INPUT_FILE} \
    --max-new-tokens 300 \
    --temperature 0.6
