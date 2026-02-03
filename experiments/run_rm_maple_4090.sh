#!/bin/bash
# 4090 服务器专用运行脚本
# 针对 Xeon CPU 低主频优化

# 设置 GPU（默认使用第7张卡）
GPU_ID=${GPU_ID:-7}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 线程设置（避免过度线程化，但不要太少）
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# 默认参数
TASK=${1:-stack}
NUM_ENVS=${2:-4}
EPOCHS=${3:-500}

echo "=============================================="
echo "4090 Server Optimized Training"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Task: $TASK"
echo "Num envs: $NUM_ENVS"
echo "Epochs: $EPOCHS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=============================================="

# 运行训练（使用优化参数）
uv run python experiments/run_rm_maple.py \
    --task $TASK \
    --num-envs $NUM_ENVS \
    --epochs $EPOCHS \
    --batch-size 2048 \
    --num-trains 500 \
    --no-video \
    "$@"
