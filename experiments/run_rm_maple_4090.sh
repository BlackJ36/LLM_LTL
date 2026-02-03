#!/bin/bash
# 4090 服务器专用运行脚本
# 针对 Xeon CPU 低主频优化

# 设置 GPU（默认使用第7张卡）
GPU_ID=${GPU_ID:-7}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 线程设置（根据 num_envs 调整）
# 8+ 环境建议用更多线程
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# torch.compile 优化：允许捕获标量输出
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# 默认参数
TASK=${1:-stack}
NUM_ENVS=${2:-4}
EPOCHS=${3:-500}

# 移除前3个位置参数，剩下的作为额外参数
shift 3 2>/dev/null || true

# 检查是否为 debug 模式
DEBUG_MODE=""
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" == "--debug" ]; then
        DEBUG_MODE="--debug"
        EPOCHS=10  # Debug 模式使用更少的 epochs
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
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
echo "torch.compile: enabled (default mode)"
echo "AMP (FP16): enabled"
echo "Target update period: 2"
echo "cuDNN benchmark: enabled"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=============================================="

if [ -n "$DEBUG_MODE" ]; then
    # Debug 模式：使用默认小参数
    uv run python experiments/run_rm_maple.py \
        --task $TASK \
        --num-envs $NUM_ENVS \
        --epochs $EPOCHS \
        --torch-compile \
        --compile-mode default \
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
        --compile-mode default \
        --amp \
        --target-update-period 2 \
        --no-video \
        $EXTRA_ARGS
fi
