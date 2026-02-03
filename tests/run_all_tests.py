#!/usr/bin/env python
"""
Run All Tests
运行所有测试脚本
"""
import subprocess
import sys
import os

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tests = [
    ("tests/test_gpu_training.py", "PyTorch GPU Training"),
    ("tests/test_mujoco_sim.py", "MuJoCo Simulation"),
    ("tests/test_maple_compat.py", "Maple/Robosuite Compatibility"),
]


def run_test(script, name):
    print("\n" + "=" * 70)
    print(f"  Running: {name}")
    print("=" * 70 + "\n")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    return result.returncode == 0


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  LLM_LTL Environment Test Suite".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    results = {}

    for script, name in tests:
        try:
            results[name] = run_test(script, name)
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = False

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  Final Summary".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"\n  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "#" * 70)
    if all_passed:
        print("\n  🎉 All tests passed! Environment is ready.\n")
        sys.exit(0)
    else:
        print("\n  ⚠️ Some tests failed. See details above.\n")
        sys.exit(1)
