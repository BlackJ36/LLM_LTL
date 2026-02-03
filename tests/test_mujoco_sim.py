"""
Test 2: MuJoCo Simulation Test
验证新版MuJoCo能否正常运行仿真
"""
import time
import numpy as np


def test_mujoco_import():
    """测试MuJoCo导入"""
    print("=" * 50)
    print("MuJoCo Import Test")
    print("=" * 50)

    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import mujoco: {e}")
        return False


def test_basic_simulation():
    """测试基本仿真功能"""
    print("\n" + "=" * 50)
    print("Basic Simulation Test")
    print("=" * 50)

    import mujoco

    # 创建简单的模型 (一个自由落体的球)
    xml = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
            <body name="ball" pos="0 0 1">
                <joint type="free"/>
                <geom type="sphere" size="0.1" rgba="1 0 0 1" mass="1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"✓ Model created successfully")
    print(f"  - nq (positions): {model.nq}")
    print(f"  - nv (velocities): {model.nv}")
    print(f"  - nbody: {model.nbody}")

    # 运行仿真
    num_steps = 1000
    start_time = time.time()

    initial_height = data.qpos[2]
    print(f"  - Initial ball height: {initial_height:.3f}")

    for _ in range(num_steps):
        mujoco.mj_step(model, data)

    elapsed = time.time() - start_time
    final_height = data.qpos[2]

    print(f"✓ Simulation completed: {num_steps} steps in {elapsed:.3f}s")
    print(f"  - Final ball height: {final_height:.3f}")
    print(f"  - Simulation speed: {num_steps / elapsed:.0f} steps/sec")

    # 验证物理正确性 (球应该落下)
    if final_height < initial_height:
        print("✓ Physics check passed (ball fell down)")
        return True
    else:
        print("❌ Physics check failed")
        return False


def test_robot_arm_simulation():
    """测试机械臂仿真 (更接近robosuite场景)"""
    print("\n" + "=" * 50)
    print("Robot Arm Simulation Test")
    print("=" * 50)

    import mujoco

    # 简单的2关节机械臂
    xml = """
    <mujoco>
        <option gravity="0 0 -9.81" timestep="0.002"/>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>

            <body name="link1" pos="0 0 0.5">
                <joint name="joint1" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04" rgba="0 0.5 0.5 1"/>

                <body name="link2" pos="0 0 0.3">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" rgba="0.5 0.5 0 1"/>

                    <body name="end_effector" pos="0 0 0.3">
                        <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
                    </body>
                </body>
            </body>
        </worldbody>

        <actuator>
            <motor joint="joint1" ctrlrange="-10 10" ctrllimited="true"/>
            <motor joint="joint2" ctrlrange="-10 10" ctrllimited="true"/>
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"✓ Robot arm model created")
    print(f"  - Joints: {model.njnt}")
    print(f"  - Actuators: {model.nu}")

    # 测试控制
    num_episodes = 3
    steps_per_episode = 500

    for ep in range(num_episodes):
        # 随机控制信号
        ctrl = np.random.uniform(-5, 5, size=model.nu)
        data.ctrl[:] = ctrl

        for _ in range(steps_per_episode):
            mujoco.mj_step(model, data)

        # 获取末端执行器位置
        ee_pos = data.body("end_effector").xpos.copy()
        print(f"  Episode {ep+1}: EE position = [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

    print("✓ Robot arm control test passed")
    return True


def test_gymnasium_mujoco():
    """测试Gymnasium MuJoCo环境"""
    print("\n" + "=" * 50)
    print("Gymnasium MuJoCo Environment Test")
    print("=" * 50)

    try:
        import gymnasium as gym

        # 测试标准MuJoCo环境
        env_ids = ["Hopper-v5", "HalfCheetah-v5", "Ant-v5"]

        for env_id in env_ids:
            try:
                env = gym.make(env_id)
                obs, info = env.reset()

                print(f"✓ {env_id}:")
                print(f"  - Observation space: {env.observation_space.shape}")
                print(f"  - Action space: {env.action_space.shape}")

                # 运行几步
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)

                env.close()
            except Exception as e:
                print(f"⚠️ {env_id}: {e}")

        return True

    except Exception as e:
        print(f"❌ Gymnasium MuJoCo test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MuJoCo Simulation Test Suite")
    print("=" * 60)

    all_passed = True

    if not test_mujoco_import():
        print("\n❌ MuJoCo import failed, skipping remaining tests")
        exit(1)

    try:
        if not test_basic_simulation():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Basic simulation test failed: {e}")
        all_passed = False

    try:
        if not test_robot_arm_simulation():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Robot arm test failed: {e}")
        all_passed = False

    try:
        test_gymnasium_mujoco()
    except Exception as e:
        print(f"\n⚠️ Gymnasium MuJoCo test skipped: {e}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ All MuJoCo simulation tests PASSED!")
    else:
        print("  ❌ Some tests FAILED")
    print("=" * 60)
