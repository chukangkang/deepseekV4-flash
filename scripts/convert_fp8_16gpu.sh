#!/bin/bash
# Convert HuggingFace DeepSeek-V4 checkpoint with 16-way model parallelism
# Usage: bash scripts/convert_fp8_16gpu.sh <hf_ckpt_path> <save_path>
#
# Non-expert weights: FP8 (float8_e4m3fn) — as stored in HF checkpoint
# Expert weights:     FP4 (float4_e2m1fn)  — kept in native FP4 format
# Shards all weights across 16 GPU ranks for 2-node x 8-GPU deployment.

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
VENV_DIR=${PROJECT_DIR}/venv_convert

HF_CKPT_PATH=${1:?"Usage: $0 <hf_ckpt_path> <save_path>"}
SAVE_PATH=${2:?"Usage: $0 <hf_ckpt_path> <save_path>"}

EXPERTS=256
MP=16

# Create venv if not exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip install --upgrade pip
    pip install torch safetensors tqdm packaging numpy
else
    echo "Using existing virtual environment at ${VENV_DIR}"
    source ${VENV_DIR}/bin/activate
fi

echo "Converting DeepSeek-V4 checkpoint (FP8 non-expert + FP4 expert)"
echo "  HF checkpoint: ${HF_CKPT_PATH}"
echo "  Save path:     ${SAVE_PATH}"
echo "  Experts:       ${EXPERTS}"
echo "  Model parallel: ${MP} (2 nodes x 8 GPUs)"
echo "  Python:        $(which python)"

python ${PROJECT_DIR}/convert.py \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --save-path ${SAVE_PATH} \
    --n-experts ${EXPERTS} \
    --model-parallel ${MP}

echo "Conversion complete. Output files:"
ls -lh ${SAVE_PATH}/model*-mp${MP}.safetensors
