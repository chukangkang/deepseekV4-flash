#!/bin/bash
# Convert HuggingFace DeepSeek-V4 checkpoint for PP=2 x TP=8
# Usage: bash scripts/convert_pp2_tp8.sh <hf_ckpt_path> <save_path>
#
# Non-expert weights: FP8 (float8_e4m3fn)
# Expert weights:     FP4 (float4_e2m1fn)
# Shards across 8 TP ranks. Each shard contains ALL layers.
# Pipeline parallelism (PP=2) is handled at runtime by loading
# only the relevant layers per stage (strict=False in load_model).

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
VENV_DIR=${PROJECT_DIR}/venv_convert

HF_CKPT_PATH=${1:?"Usage: $0 <hf_ckpt_path> <save_path>"}
SAVE_PATH=${2:?"Usage: $0 <hf_ckpt_path> <save_path>"}

EXPERTS=256
MP=8

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

echo "Converting DeepSeek-V4 checkpoint for PP=2 x TP=8"
echo "  HF checkpoint: ${HF_CKPT_PATH}"
echo "  Save path:     ${SAVE_PATH}"
echo "  Experts:       ${EXPERTS}"
echo "  Model parallel: ${MP} (TP=8, PP handled at runtime)"
echo "  Python:        $(which python3)"

python3 ${PROJECT_DIR}/convert.py \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --save-path ${SAVE_PATH} \
    --n-experts ${EXPERTS} \
    --model-parallel ${MP}

echo ""
echo "Conversion complete. Output files:"
ls -lh ${SAVE_PATH}/model*-mp${MP}.safetensors
echo ""
echo "Copy these files + tokenizer to both nodes."
echo "Each node loads the same shards; PP stage selection happens at runtime."
