#!/usr/bin/env python3
"""
Performance comparison: Single environment vs SubprocVecEnv (4 environments)
Runs 20 epochs for each configuration and compares results.
"""
import subprocess
import sys
import time
import os

os.chdir("/home/fxxk/git/LLM_LTL")

def run_experiment(num_envs: int, label: str):
    """Run training experiment and capture results."""
    print(f"\n{'='*70}")
    print(f"Running: {label} (num_envs={num_envs})")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable, "maple/scripts/train.py",
        "--env", "lift",
        "--num_envs", str(num_envs),
        "--label", label,
        "--no_video",
        "--debug",      # Use debug mode for faster iterations
        "--epochs", "20"  # Override to 20 epochs
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    wall_time = end_time - start_time

    # Print last portion of output
    output = result.stdout
    if len(output) > 8000:
        output = "...[truncated]...\n" + output[-8000:]
    print(output)

    if result.returncode != 0:
        print("STDERR (last 2000 chars):")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    print(f"\n>>> Wall time for {label}: {wall_time:.2f}s <<<")

    return {
        'label': label,
        'num_envs': num_envs,
        'wall_time': wall_time,
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def extract_metrics(stdout: str):
    """Extract key metrics from training output."""
    metrics = {}

    # Look for final epoch metrics
    lines = stdout.split('\n')
    for line in reversed(lines):
        if 'epoch' in line.lower() and 'return' in line.lower():
            # Try to parse metrics
            break

    # Extract exploration/evaluation returns
    for line in lines:
        if 'expl/Average Returns' in line or 'exploration/Average Returns' in line:
            try:
                val = float(line.split(':')[-1].strip())
                metrics['expl_return'] = val
            except:
                pass
        if 'eval/Average Returns' in line or 'evaluation/Average Returns' in line:
            try:
                val = float(line.split(':')[-1].strip())
                metrics['eval_return'] = val
            except:
                pass

    return metrics

if __name__ == "__main__":
    print("="*70)
    print("MAPLE Vectorized Environment Performance Comparison")
    print("="*70)
    print(f"Test: Single env vs SubprocVecEnv (4 envs), 20 epochs each")
    print("="*70)

    # Run single environment experiment
    result_single = run_experiment(num_envs=1, label="single_env_20ep")

    # Run vectorized environment experiment
    result_vec = run_experiment(num_envs=4, label="vec_env_20ep")

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Configuration':<30} {'Wall Time':<15} {'Status'}")
    print("-"*60)
    print(f"{'Single env (num_envs=1)':<30} {result_single['wall_time']:>10.2f}s    {'OK' if result_single['returncode'] == 0 else 'FAILED'}")
    print(f"{'SubprocVecEnv (num_envs=4)':<30} {result_vec['wall_time']:>10.2f}s    {'OK' if result_vec['returncode'] == 0 else 'FAILED'}")

    if result_single['wall_time'] > 0 and result_vec['wall_time'] > 0:
        speedup = result_single['wall_time'] / result_vec['wall_time']
        print(f"\nSpeedup: {speedup:.2f}x")

        if speedup > 1:
            time_saved = result_single['wall_time'] - result_vec['wall_time']
            print(f"Time saved: {time_saved:.2f}s ({(1 - 1/speedup)*100:.1f}%)")

    print("\n" + "="*70)
