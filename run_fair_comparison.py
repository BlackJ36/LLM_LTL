#!/usr/bin/env python3
"""
Fair comparison with longer episodes to reduce multiprocessing overhead ratio.
"""
import subprocess
import sys
import time
import os

os.chdir("/home/fxxk/git/LLM_LTL")

def run_experiment(num_envs: int, label: str, num_steps: int = 500):
    """Run training experiment with specified steps."""
    print(f"\n{'='*70}")
    print(f"Running: {label} (num_envs={num_envs}, steps_per_loop={num_steps})")
    print(f"{'='*70}\n")

    # Use non-debug mode but with reduced epochs for time
    cmd = [
        sys.executable, "maple/scripts/train.py",
        "--env", "lift",
        "--num_envs", str(num_envs),
        "--label", label,
        "--no_video",
        "--epochs", "5",  # Just 5 epochs for testing
        "--snapshot_gap", "100",  # Avoid saving snapshots
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    end_time = time.time()
    wall_time = end_time - start_time

    # Extract timing info from last epoch
    lines = result.stdout.split('\n')
    timing_info = {}
    for line in lines:
        if 'time/exploration sampling' in line:
            try:
                timing_info['expl_sample'] = float(line.split()[-1])
            except: pass
        elif 'time/epoch' in line:
            try:
                timing_info['epoch'] = float(line.split()[-1])
            except: pass

    # Print last portion of output
    output = result.stdout
    if len(output) > 5000:
        output = "...[truncated]...\n" + output[-5000:]
    print(output)

    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr[-1500:])

    print(f"\n>>> Wall time for {label}: {wall_time:.2f}s <<<")
    if timing_info:
        print(f">>> Exploration sampling per epoch: {timing_info.get('expl_sample', 'N/A')}s <<<")
        print(f">>> Time per epoch: {timing_info.get('epoch', 'N/A')}s <<<")

    return {
        'label': label,
        'wall_time': wall_time,
        'timing': timing_info
    }

if __name__ == "__main__":
    print("="*70)
    print("MAPLE Fair Performance Comparison (Non-Debug Mode)")
    print("="*70)
    print("3000 exploration steps per loop, 150 max path length, 5 epochs")
    print("="*70)

    # Run single environment
    result_single = run_experiment(num_envs=1, label="single_real")

    # Run vectorized environment
    result_vec = run_experiment(num_envs=4, label="vec_real")

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY (Non-Debug Mode)")
    print("="*70)

    print(f"\n{'Configuration':<35} {'Wall Time':<12} {'Expl/Epoch':<12}")
    print("-"*65)

    expl1 = result_single['timing'].get('expl_sample', 'N/A')
    expl2 = result_vec['timing'].get('expl_sample', 'N/A')

    print(f"{'Single env (num_envs=1)':<35} {result_single['wall_time']:>8.2f}s   {expl1}")
    print(f"{'SubprocVecEnv (num_envs=4)':<35} {result_vec['wall_time']:>8.2f}s   {expl2}")

    if result_single['wall_time'] > 0 and result_vec['wall_time'] > 0:
        speedup = result_single['wall_time'] / result_vec['wall_time']
        print(f"\nOverall Speedup: {speedup:.2f}x")

        if isinstance(expl1, float) and isinstance(expl2, float):
            expl_speedup = expl1 / expl2
            print(f"Exploration Sampling Speedup: {expl_speedup:.2f}x")

    print("\n" + "="*70)
