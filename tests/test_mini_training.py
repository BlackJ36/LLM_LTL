"""
Mini Training Test
小规模训练测试，验证完整训练流程是否正常工作
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import torch
import time


class DummyEnv:
    """用于测试的虚拟环境"""
    def __init__(self, obs_dim=32, action_dim=8):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @property
    def observation_space(self):
        class Space:
            def __init__(self, dim):
                self.shape = (dim,)
                self.low = -np.inf * np.ones(dim)
                self.high = np.inf * np.ones(dim)
        return Space(self.obs_dim)

    @property
    def action_space(self):
        class Space:
            def __init__(self, dim):
                self.shape = (dim,)
                self.low = -np.ones(dim)
                self.high = np.ones(dim)
        return Space(self.action_dim)


def test_sac_training_loop():
    """测试完整的SAC训练循环"""
    print("=" * 60)
    print("  SAC Training Loop Test")
    print("=" * 60)

    from maple.torch.sac.sac import SACTrainer
    from maple.torch.networks.mlp import ConcatMlp
    from maple.torch.sac.policies import TanhGaussianPolicy
    from maple.data_management.simple_replay_buffer import SimpleReplayBuffer
    import maple.torch.pytorch_util as ptu

    # 设置GPU模式 - 这对于ptu内部的tensor创建很重要
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        ptu.set_gpu_mode(True, 0)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # 环境参数
    obs_dim = 32
    action_dim = 8

    # 创建虚拟环境
    dummy_env = DummyEnv(obs_dim, action_dim)

    # 创建网络
    hidden_sizes = [256, 256]

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    ).to(device)

    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    ).to(device)

    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    ).to(device)

    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    ).to(device)

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)

    print("✓ Networks created")

    # 创建Trainer
    trainer = SACTrainer(
        env=dummy_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        discount=0.99,
        soft_target_tau=0.005,
        policy_lr=3e-4,
        qf_lr=3e-4,
    )
    print("✓ SACTrainer created")

    # 创建Replay Buffer
    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=10000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        env_info_sizes={},
    )
    print("✓ ReplayBuffer created")

    # 模拟数据收集
    print("\nSimulating data collection...")
    num_samples = 2000
    for i in range(num_samples):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        reward = np.array([np.random.randn()], dtype=np.float32)
        terminal = np.array([0.0], dtype=np.float32)

        replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward[0],
            terminal=terminal[0],
            next_observation=next_obs,
            env_info={},
        )

    print(f"✓ Added {num_samples} samples to replay buffer")

    # 执行训练
    print("\nRunning training updates...")
    batch_size = 256
    num_updates = 100

    start_time = time.time()
    losses = []

    for i in range(num_updates):
        batch = replay_buffer.random_batch(batch_size)

        # 转换为torch tensor并移到GPU (确保float32类型)
        batch_torch = {
            'observations': torch.from_numpy(batch['observations']).float().to(device),
            'actions': torch.from_numpy(batch['actions']).float().to(device),
            'rewards': torch.from_numpy(batch['rewards']).float().to(device),
            'terminals': torch.from_numpy(batch['terminals']).float().to(device),
            'next_observations': torch.from_numpy(batch['next_observations']).float().to(device),
        }

        trainer.train_from_torch(batch_torch)

        if (i + 1) % 20 == 0:
            stats = trainer.get_diagnostics()
            qf_loss = stats.get('QF1 Loss', 0)
            policy_loss = stats.get('Policy Loss', 0)
            print(f"  Update {i+1}/{num_updates}: QF Loss={qf_loss:.4f}, Policy Loss={policy_loss:.4f}")
            losses.append(qf_loss)

    elapsed = time.time() - start_time
    print(f"\n✓ Training completed: {num_updates} updates in {elapsed:.2f}s")
    print(f"✓ Updates per second: {num_updates / elapsed:.1f}")

    return True


def test_robosuite_env():
    """测试Robosuite环境"""
    print("\n" + "=" * 60)
    print("  Robosuite Environment Test")
    print("=" * 60)

    try:
        import robosuite as suite

        # maple分支需要skill_config
        skill_config = dict(
            skills=['atomic', 'reach', 'grasp', 'push'],
            aff_penalty_fac=15.0,
            base_config=dict(
                global_xyz_bounds=[[-0.30, -0.30, 0.80], [0.15, 0.30, 0.95]],
                lift_height=0.95,
                binary_gripper=True,
                aff_threshold=0.06,
                aff_type='dense',
                aff_tanh_scaling=10.0,
            ),
            atomic_config=dict(use_ori_params=True),
            reach_config=dict(
                use_gripper_params=False,
                local_xyz_scale=[0.0, 0.0, 0.06],
                use_ori_params=False,
                max_ac_calls=15,
            ),
            grasp_config=dict(
                global_xyz_bounds=[[-0.30, -0.30, 0.80], [0.15, 0.30, 0.85]],
                aff_threshold=0.03,
                local_xyz_scale=[0.0, 0.0, 0.0],
                use_ori_params=True,
                max_ac_calls=20,
                num_reach_steps=2,
                num_grasp_steps=3,
            ),
            push_config=dict(
                global_xyz_bounds=[[-0.30, -0.30, 0.80], [0.15, 0.30, 0.85]],
                delta_xyz_scale=[0.25, 0.25, 0.05],
                max_ac_calls=20,
                use_ori_params=True,
                aff_threshold=[0.12, 0.12, 0.04],
            ),
        )

        # 创建简单的Lift环境
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            skill_config=skill_config,
        )

        print(f"✓ Environment created: Lift with Panda robot")
        print(f"  - Observation space dim: {env.observation_spec()}")
        action_low, action_high = env.action_spec
        action_dim = len(action_low)
        print(f"  - Action space dim: {action_dim}")

        # 运行几步
        obs = env.reset()
        print(f"✓ Environment reset, obs keys: {list(obs.keys())}")

        total_reward = 0
        for step in range(10):
            action = np.random.uniform(-1, 1, action_dim)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"✓ Ran 10 steps, total reward: {total_reward:.4f}")

        env.close()
        print("✓ Environment closed")

        return True

    except Exception as e:
        print(f"❌ Robosuite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_maple_with_robosuite():
    """测试Maple与Robosuite的集成"""
    print("\n" + "=" * 60)
    print("  Maple + Robosuite Integration Test")
    print("=" * 60)

    try:
        import robosuite as suite
        from maple.torch.sac.policies import TanhGaussianPolicy
        from maple.torch.networks.mlp import Mlp
        import maple.torch.pytorch_util as ptu

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            ptu.set_gpu_mode(True, 0)
        device = torch.device("cuda" if use_cuda else "cpu")

        # maple分支需要skill_config
        skill_config = dict(
            skills=['atomic', 'reach', 'grasp', 'push'],
            aff_penalty_fac=15.0,
            base_config=dict(
                global_xyz_bounds=[[-0.30, -0.30, 0.80], [0.15, 0.30, 0.95]],
                lift_height=0.95,
                binary_gripper=True,
                aff_threshold=0.06,
                aff_type='dense',
                aff_tanh_scaling=10.0,
            ),
            atomic_config=dict(use_ori_params=True),
            reach_config=dict(
                use_gripper_params=False,
                local_xyz_scale=[0.0, 0.0, 0.06],
                use_ori_params=False,
                max_ac_calls=15,
            ),
            grasp_config=dict(
                global_xyz_bounds=[[-0.30, -0.30, 0.80], [0.15, 0.30, 0.85]],
                aff_threshold=0.03,
                local_xyz_scale=[0.0, 0.0, 0.0],
                use_ori_params=True,
                max_ac_calls=20,
                num_reach_steps=2,
                num_grasp_steps=3,
            ),
            push_config=dict(
                global_xyz_bounds=[[-0.30, -0.30, 0.80], [0.15, 0.30, 0.85]],
                delta_xyz_scale=[0.25, 0.25, 0.05],
                max_ac_calls=20,
                use_ori_params=True,
                aff_threshold=[0.12, 0.12, 0.04],
            ),
        )

        # 创建环境
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            skill_config=skill_config,
        )

        # 获取维度
        obs = env.reset()
        obs_dim = sum([v.shape[0] if len(v.shape) > 0 else 1 for v in obs.values()])
        action_low, action_high = env.action_spec
        action_dim = len(action_low)

        print(f"✓ Env dimensions: obs={obs_dim}, action={action_dim}")

        # 创建策略网络
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[256, 256],
        ).to(device)

        print("✓ Policy network created")

        # 使用策略进行动作采样
        obs_flat = np.concatenate([v.flatten() for v in obs.values()])
        obs_tensor = torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)

        with torch.no_grad():
            dist = policy(obs_tensor)
            action, log_prob = dist.rsample_and_logprob()

        print(f"✓ Policy forward pass: action shape={action.shape}")

        # 运行一个episode
        total_reward = 0
        obs = env.reset()
        for step in range(50):
            obs_flat = np.concatenate([v.flatten() for v in obs.values()])
            obs_tensor = torch.from_numpy(obs_flat).float().unsqueeze(0).to(device)

            with torch.no_grad():
                dist = policy(obs_tensor)
                action = dist.sample()

            action_np = action.cpu().numpy().flatten()
            obs, reward, done, info = env.step(action_np)
            total_reward += reward

            if done:
                break

        print(f"✓ Ran episode: {step+1} steps, total reward: {total_reward:.4f}")

        env.close()
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#" + "  Mini Training Test Suite".center(58) + "#")
    print("#" * 60 + "\n")

    results = {}

    # 测试SAC训练循环
    try:
        results["sac_training"] = test_sac_training_loop()
    except Exception as e:
        print(f"❌ SAC training test error: {e}")
        import traceback
        traceback.print_exc()
        results["sac_training"] = False

    # 测试Robosuite环境
    try:
        results["robosuite_env"] = test_robosuite_env()
    except Exception as e:
        print(f"❌ Robosuite env test error: {e}")
        import traceback
        traceback.print_exc()
        results["robosuite_env"] = False

    # 测试集成
    try:
        results["integration"] = test_maple_with_robosuite()
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        results["integration"] = False

    # 总结
    print("\n" + "#" * 60)
    print("#" + "  Test Summary".center(58) + "#")
    print("#" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 All mini training tests passed!")
        print("  The training pipeline is ready to use.")
    else:
        print("  ⚠️ Some tests failed. Check the errors above.")

    print("#" * 60 + "\n")
