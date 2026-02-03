#!/usr/bin/env python
"""
并行训练脚本 - 在多个GPU上同时运行多个实验
"""
import argparse
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def run_experiment(args):
    """运行单个实验"""
    env, seed, gpu_id, label, debug = args

    cmd = [
        "uv", "run", "python", "maple/scripts/train.py",
        "--env", env,
        "--label", f"{label}_seed{seed}",
        "--gpu_id", str(gpu_id),
    ]

    if debug:
        cmd.append("--debug")

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] 启动: {env} seed={seed}")
    start_time = time.time()

    result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)

    elapsed = time.time() - start_time
    status = "成功" if result.returncode == 0 else "失败"
    print(f"[GPU {gpu_id}] {status}: {env} seed={seed} ({elapsed:.1f}s)")

    return {
        "env": env,
        "seed": seed,
        "gpu_id": gpu_id,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "stdout": result.stdout[-500:] if result.stdout else "",
        "stderr": result.stderr[-500:] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser(description="并行训练MAPLE")
    parser.add_argument("--envs", nargs="+", default=["lift"],
                       choices=["lift", "door", "pnp", "wipe", "stack", "nut", "cleanup", "peg_ins"],
                       help="要训练的环境列表")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                       help="随机种子列表")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                       help="可用的GPU ID列表")
    parser.add_argument("--label", type=str, default="parallel",
                       help="实验标签")
    parser.add_argument("--debug", action="store_true",
                       help="Debug模式(少量epochs)")
    parser.add_argument("--max_parallel", type=int, default=None,
                       help="最大并行数(默认=GPU数量)")
    args = parser.parse_args()

    # 生成所有实验组合
    experiments = []
    gpu_cycle = 0
    for env in args.envs:
        for seed in args.seeds:
            gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
            experiments.append((env, seed, gpu_id, args.label, args.debug))
            gpu_cycle += 1

    max_workers = args.max_parallel or len(args.gpus)
    print(f"总共 {len(experiments)} 个实验, 使用 {max_workers} 个并行进程")
    print(f"GPUs: {args.gpus}")
    print("-" * 50)

    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in experiments}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # 打印总结
    print("\n" + "=" * 50)
    print("训练总结:")
    print("=" * 50)

    success = sum(1 for r in results if r["returncode"] == 0)
    failed = len(results) - success

    for r in results:
        status = "✓" if r["returncode"] == 0 else "✗"
        print(f"  {status} {r['env']} seed={r['seed']} GPU={r['gpu_id']} ({r['elapsed']:.1f}s)")

    print(f"\n成功: {success}/{len(results)}, 失败: {failed}/{len(results)}")


if __name__ == "__main__":
    main()
