"""
Test 3: Maple/Robosuite Compatibility Test
验证maple和robosuite在新环境中的兼容性
"""
import sys


def test_maple_imports():
    """测试maple核心模块导入"""
    print("=" * 50)
    print("Maple Core Imports Test")
    print("=" * 50)

    modules_to_test = [
        ("maple", "Main package"),
        ("maple.torch.pytorch_util", "PyTorch utilities"),
        ("maple.torch.networks.mlp", "MLP networks"),
        ("maple.torch.distributions", "Distributions"),
        ("maple.torch.sac.sac", "SAC trainer"),
        ("maple.torch.sac.sac_hybrid", "SAC Hybrid trainer"),
        ("maple.torch.sac.policies", "Policies"),
        ("maple.data_management.replay_buffer", "Replay buffer"),
        ("maple.samplers.data_collector", "Data collector"),
    ]

    all_passed = True
    for module, desc in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}: {desc}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            all_passed = False

    return all_passed


def test_robosuite_imports():
    """测试robosuite核心模块导入"""
    print("\n" + "=" * 50)
    print("Robosuite Core Imports Test")
    print("=" * 50)

    modules_to_test = [
        ("robosuite", "Main package"),
        ("robosuite.environments", "Environments"),
        ("robosuite.robots", "Robots"),
        ("robosuite.controllers", "Controllers"),
    ]

    all_passed = True
    for module, desc in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}: {desc}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            all_passed = False

    return all_passed


def test_pytorch_components():
    """测试maple的PyTorch组件"""
    print("\n" + "=" * 50)
    print("Maple PyTorch Components Test")
    print("=" * 50)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 测试MLP网络
        from maple.torch.networks.mlp import Mlp

        mlp = Mlp(
            input_size=32,
            output_size=10,
            hidden_sizes=[256, 256]
        ).to(device)

        x = torch.randn(64, 32, device=device)
        y = mlp(x)
        print(f"✓ Mlp: input {x.shape} -> output {y.shape}")
    except Exception as e:
        print(f"❌ Mlp test failed: {e}")
        return False

    try:
        # 测试分布
        from maple.torch.distributions import TanhNormal

        mean = torch.randn(64, 8, device=device)
        std = torch.ones(64, 8, device=device) * 0.5
        dist = TanhNormal(mean, std)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        print(f"✓ TanhNormal: sample {sample.shape}, log_prob {log_prob.shape}")
    except Exception as e:
        print(f"❌ TanhNormal test failed: {e}")
        return False

    try:
        # 测试SAC Trainer组件
        from maple.torch.sac.sac import SACTrainer

        print("✓ SACTrainer imported successfully")
    except Exception as e:
        print(f"❌ SACTrainer import failed: {e}")
        return False

    return True


def test_gym_compatibility():
    """测试gym/gymnasium兼容性"""
    print("\n" + "=" * 50)
    print("Gym/Gymnasium Compatibility Test")
    print("=" * 50)

    # 检查maple是否使用了旧的gym API
    issues = []

    # 检查是否有gym导入
    try:
        import gym
        print("⚠️ Old 'gym' package is importable (maple may use it)")
    except ImportError:
        print("✓ Old 'gym' not installed (good)")

    try:
        import gymnasium
        print(f"✓ gymnasium version: {gymnasium.__version__}")
    except ImportError:
        print("❌ gymnasium not installed")
        return False

    # 检查maple的gym使用
    try:
        # 查找maple中的gym引用
        import maple.envs
        print("✓ maple.envs imported")
    except Exception as e:
        print(f"⚠️ maple.envs: {e}")
        issues.append("maple.envs may need gym compatibility updates")

    if issues:
        print("\n⚠️ Potential compatibility issues:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def test_mujoco_api_compatibility():
    """测试MuJoCo API兼容性"""
    print("\n" + "=" * 50)
    print("MuJoCo API Compatibility Test")
    print("=" * 50)

    # 新版mujoco vs mujoco-py API差异
    print("Checking for mujoco-py style imports in robosuite...")

    import os
    import re
    import robosuite

    # 处理namespace package的情况
    if hasattr(robosuite, '__path__'):
        robosuite_path = robosuite.__path__[0]
    elif hasattr(robosuite, '__file__') and robosuite.__file__:
        robosuite_path = os.path.dirname(robosuite.__file__)
    else:
        print("⚠️ Could not determine robosuite path")
        return True

    mujoco_py_imports = []

    for root, dirs, files in os.walk(robosuite_path):
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r") as file:
                        content = file.read()
                        if "mujoco_py" in content or "import mujoco_py" in content:
                            rel_path = os.path.relpath(filepath, robosuite_path)
                            mujoco_py_imports.append(rel_path)
                except:
                    pass

    if mujoco_py_imports:
        print(f"⚠️ Found {len(mujoco_py_imports)} files with mujoco_py references:")
        for f in mujoco_py_imports[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(mujoco_py_imports) > 10:
            print(f"  ... and {len(mujoco_py_imports) - 10} more")
        print("\n  These files need to be updated to use new mujoco API")
        return False
    else:
        print("✓ No mujoco_py imports found in robosuite")
        return True


def summary_and_recommendations():
    """总结和建议"""
    print("\n" + "=" * 60)
    print("  Summary and Recommendations")
    print("=" * 60)

    print("""
If you see mujoco_py compatibility issues, you have two options:

Option 1: Use robosuite's official v1.4+ (recommended)
  - robosuite v1.4+ supports new mujoco package
  - git clone https://github.com/ARISE-Initiative/robosuite
  - git checkout v1.4.1

Option 2: Create compatibility shim
  - Create a mujoco_py compatibility layer
  - Map old API calls to new mujoco API

Option 3: Use the MAPLE-compatible robosuite branch
  - The maple branch might need updates for new mujoco
  - Consider contributing the updates back
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Maple/Robosuite Compatibility Test Suite")
    print("=" * 60)

    results = {}

    results["maple_imports"] = test_maple_imports()
    results["robosuite_imports"] = test_robosuite_imports()
    results["pytorch_components"] = test_pytorch_components()
    results["gym_compat"] = test_gym_compatibility()
    results["mujoco_api"] = test_mujoco_api_compatibility()

    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if not all_passed:
        summary_and_recommendations()

    print("\n" + "=" * 60)
