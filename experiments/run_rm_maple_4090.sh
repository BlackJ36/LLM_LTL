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

# 检查是否为 debug 模式
DEBUG_MODE=""
for arg in "$@"; do
    if [ "$arg" == "--debug" ]; then
        DEBUG_MODE="--debug"
        EPOCHS=10  # Debug 模式使用更少的 epochs
        break
    fi
done

echo "=============================================="
echo "4090 Server Optimized Training"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Task: $TASK"
echo "Num envs: $NUM_ENVS"
echo "Epochs: $EPOCHS"
if [ -n "$DEBUG_MODE" ]; then
    echo "Mode: DEBUG (reduced steps)"
else
    echo "Batch size: 2048 (2x default)"
    echo "Num trains: 500 (0.5x default)"
    echo "LR scale: 2.0 (linear scaling)"
fi
echo "torch.compile: enabled (reduce-overhead)"
echo "AMP (FP16): enabled"
echo "Target update period: 2"
echo "cuDNN benchmark: enabled"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=============================================="

# 过滤掉已处理的参数
EXTRA_ARGS=""
for arg in "$@"; do
    case $arg in
        --debug) ;;  # 已处理
        *) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

if [ -n "$DEBUG_MODE" ]; then
    # Debug 模式：使用默认小参数
    uv run python experiments/run_rm_maple.py \
        --task $TASK \
        --num-envs $NUM_ENVS \
        --epochs $EPOCHS \
        --torch-compile \
        --amp \
        --target-update-period 2 \
        --no-video \
        --debug \
        $EXTRA_ARGS
else
    # 正常模式：使用优化参数
    uv run python experiments/run_rm_maple.py \
        --task $TASK \
        --num-envs $NUM_ENVS \
        --epochs $EPOCHS \
        --batch-size 2048 \
        --num-trains 500 \
        --lr-scale 2.0 \
        --torch-compile \
        --amp \
        --target-update-period 2 \
        --no-video \
        $EXTRA_ARGS
fi
