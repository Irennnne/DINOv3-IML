#!/usr/bin/env bash
# DINOv3-IML training launcher
#
# Usage:
#   bash scripts/train.sh configs/lora_vitl_r32.yaml
#
# Requirements:
#   pip install imdlbenco peft torch
#
# Before running:
#   1. Edit the config YAML to set data_path, test_data_path,
#      dinov3_repo_path, and dinov3_weights_path for your system.
#   2. Ensure the DINOv3 backbone weights are downloaded.
#
set -euo pipefail

CONFIG="${1:-configs/lora_vitl_r32.yaml}"
CONFIG_TOOL="scripts/resolve_config.py"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CONFIG_TOOL" ]; then
    echo "Error: config resolver not found: $CONFIG_TOOL"
    exit 1
fi

MODEL=$(python3 "$CONFIG_TOOL" "$CONFIG" --field model)
OUTPUT_DIR="output/$(basename ${CONFIG%.yaml})"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC="${NPROC:-1}"

echo "========================================"
echo "Config:     $CONFIG"
echo "Model:      $MODEL"
echo "Output:     $OUTPUT_DIR"
echo "GPUs:       $NPROC"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# Local training entry point registers this repo's custom models first.
torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    train.py \
    $(python3 "$CONFIG_TOOL" "$CONFIG" --cli-args) \
    --model "$MODEL" \
    --data_path "$(python3 "$CONFIG_TOOL" "$CONFIG" --field data_path)" \
    --test_data_path "$(python3 "$CONFIG_TOOL" "$CONFIG" --field test_data_path)" \
    --dinov3_repo_path "$(python3 "$CONFIG_TOOL" "$CONFIG" --field dinov3_repo_path)" \
    --dinov3_weights_path "$(python3 "$CONFIG_TOOL" "$CONFIG" --field dinov3_weights_path)" \
    --output_dir "$OUTPUT_DIR" \
    --find_unused_parameters \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training complete. Results in: $OUTPUT_DIR"
