#!/bin/bash
# 并行训练脚本 - 使用 GNU parallel 或 tmux

# 默认参数
ENV="${1:-lift}"
SEEDS="${2:-0 1 2}"
LABEL="${3:-parallel}"
DEBUG="${4:-}"  # 传入 "--debug" 启用debug模式

# 检查是否安装了 parallel
if command -v parallel &> /dev/null; then
    echo "使用 GNU parallel 运行实验..."
    echo "环境: $ENV, Seeds: $SEEDS, Label: $LABEL"

    # 生成命令列表并并行执行
    for seed in $SEEDS; do
        echo "uv run python maple/scripts/train.py --env $ENV --label ${LABEL}_s${seed} $DEBUG"
    done | parallel -j 2 --progress

elif command -v tmux &> /dev/null; then
    echo "使用 tmux 运行实验..."

    SESSION="maple_train"
    tmux new-session -d -s $SESSION 2>/dev/null || true

    window=0
    for seed in $SEEDS; do
        tmux new-window -t $SESSION:$window -n "seed$seed" \
            "uv run python maple/scripts/train.py --env $ENV --label ${LABEL}_s${seed} $DEBUG; read -p 'Press Enter...'"
        ((window++))
    done

    echo "实验已在 tmux 会话 '$SESSION' 中启动"
    echo "查看: tmux attach -t $SESSION"

else
    echo "未找到 parallel 或 tmux，顺序执行..."
    for seed in $SEEDS; do
        echo "=== Seed $seed ==="
        uv run python maple/scripts/train.py --env $ENV --label ${LABEL}_s${seed} $DEBUG
    done
fi
