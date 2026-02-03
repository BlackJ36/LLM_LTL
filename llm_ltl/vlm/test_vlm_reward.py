"""
Test script for VLM reward wrapper integration with MAPLE.

Usage:
    python -m llm_ltl.vlm.test_vlm_reward --task Lift --test-vlm-only
    python -m llm_ltl.vlm.test_vlm_reward --task Lift --full-rollout
"""

import argparse
import numpy as np
import time

def test_vlm_client_connection(api_base: str):
    """Test basic connection to the VLM API."""
    print("=" * 60)
    print("Testing VLM Client Connection")
    print("=" * 60)

    from llm_ltl.vlm import QwenVLClient

    client = QwenVLClient(api_base=api_base)

    # Create a simple test image (red square)
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    test_image[64:192, 64:192, 0] = 255  # Red square

    print(f"API endpoint: {client.endpoint}")
    print("Sending test request...")

    start_time = time.time()
    is_reasonable, confidence, reason = client.evaluate_action(
        image=test_image,
        task_description="Pick up the red cube and place it on the green target",
        selected_primitive="reach",
        available_primitives=["reach", "grasp", "lift", "place", "release"],
    )
    elapsed = time.time() - start_time

    print(f"\nResponse received in {elapsed:.2f}s:")
    print(f"  Reasonable: {is_reasonable}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Reason: {reason}")

    print("\nTesting progress evaluation...")
    start_time = time.time()
    progress, stage, next_step = client.evaluate_progress(
        image=test_image,
        task_description="Pick up the red cube and place it on the green target",
    )
    elapsed = time.time() - start_time

    print(f"\nResponse received in {elapsed:.2f}s:")
    print(f"  Progress: {progress:.2f}")
    print(f"  Stage: {stage}")
    print(f"  Next step: {next_step}")

    print("\n✓ VLM client connection test passed!")
    return True


def test_with_robosuite(task: str, api_base: str, num_steps: int = 10):
    """Test VLM reward wrapper with a robosuite environment."""
    print("=" * 60)
    print(f"Testing VLM Reward Wrapper with {task}")
    print("=" * 60)

    import robosuite
    from llm_ltl.vlm import QwenVLClient, VLMRewardWrapper

    # Create environment
    print("\nCreating robosuite environment...")
    env = robosuite.make(
        task,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
    )

    # Create VLM client
    vlm_client = QwenVLClient(api_base=api_base)

    # Create a mock skill controller for testing
    class MockSkillController:
        skill_names = ["reach", "grasp", "lift", "move", "place", "release"]

        def get_skill_name_from_action(self, action):
            # Simple mock: just return first skill
            return self.skill_names[0]

    skill_controller = MockSkillController()

    # Wrap environment
    print("Wrapping environment with VLM reward...")
    wrapped_env = VLMRewardWrapper(
        env=env,
        task_description="Pick up the cube and lift it",
        skill_controller=skill_controller,
        vlm_client=vlm_client,
        vlm_reward_scale=0.5,
        eval_frequency=1,  # Evaluate every step
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
    )

    # Run test rollout
    print(f"\nRunning {num_steps} step test rollout...")
    obs = wrapped_env.reset()

    total_env_reward = 0
    total_vlm_reward = 0

    for step in range(num_steps):
        # Random action
        action = np.random.randn(env.action_dim)
        action = np.clip(action, -1, 1)

        obs, reward, done, info = wrapped_env.step(action)

        env_reward = info.get("env_reward", 0)
        vlm_reward = info.get("vlm_reward_scaled", 0)
        total_env_reward += env_reward
        total_vlm_reward += vlm_reward

        # Print info if VLM was evaluated
        if "vlm_binary_reasonable" in info:
            print(f"\nStep {step + 1}:")
            print(f"  Env reward: {env_reward:.4f}")
            print(f"  VLM reward: {vlm_reward:.4f}")
            print(f"  Total reward: {reward:.4f}")
            print(f"  VLM judgment: {'✓' if info['vlm_binary_reasonable'] else '✗'} "
                  f"(confidence: {info['vlm_binary_confidence']:.2f})")
            print(f"  Progress: {info['vlm_progress']:.2f}")

        if done:
            print("\nEpisode done!")
            break

    print(f"\n{'=' * 40}")
    print(f"Total env reward: {total_env_reward:.4f}")
    print(f"Total VLM reward: {total_vlm_reward:.4f}")
    print(f"Combined total: {total_env_reward + total_vlm_reward:.4f}")

    # Cleanup
    env.close()

    print("\n✓ VLM reward wrapper test passed!")
    return True


def test_async_wrapper(task: str, api_base: str, num_steps: int = 20):
    """Test the async VLM reward wrapper."""
    print("=" * 60)
    print("Testing Async VLM Reward Wrapper")
    print("=" * 60)

    import robosuite
    from llm_ltl.vlm import QwenVLClient
    from llm_ltl.vlm.vlm_reward_wrapper import VLMRewardWrapperAsync

    # Create environment
    print("\nCreating robosuite environment...")
    env = robosuite.make(
        task,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
    )

    # Create VLM client
    vlm_client = QwenVLClient(api_base=api_base)

    # Mock skill controller
    class MockSkillController:
        skill_names = ["reach", "grasp", "lift", "move", "place", "release"]
        def get_skill_name_from_action(self, action):
            return self.skill_names[0]

    # Wrap with async wrapper
    print("Wrapping environment with ASYNC VLM reward...")
    wrapped_env = VLMRewardWrapperAsync(
        env=env,
        task_description="Pick up the cube and lift it",
        skill_controller=MockSkillController(),
        vlm_client=vlm_client,
        vlm_reward_scale=0.5,
        eval_frequency=1,  # Every step
        camera_name="frontview",
    )

    # Run test
    print(f"\nRunning {num_steps} step async test...")
    obs = wrapped_env.reset()

    step_times = []

    for step in range(num_steps):
        start = time.time()

        action = np.random.randn(env.action_dim)
        action = np.clip(action, -1, 1)
        obs, reward, done, info = wrapped_env.step(action)

        elapsed = time.time() - start
        step_times.append(elapsed)

        if "vlm_binary_reasonable" in info:
            print(f"Step {step + 1}: reward={reward:.4f}, "
                  f"vlm_progress={info.get('vlm_progress', 0):.2f}, "
                  f"step_time={elapsed*1000:.1f}ms")

        if done:
            break

    avg_step_time = np.mean(step_times)
    print(f"\nAverage step time: {avg_step_time*1000:.1f}ms")
    print("(Async wrapper should have minimal impact on step time)")

    wrapped_env.close()

    print("\n✓ Async wrapper test passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test VLM reward integration")
    parser.add_argument("--api-base", type=str, default="http://172.19.1.40:8001",
                        help="VLM API base URL")
    parser.add_argument("--task", type=str, default="Lift",
                        help="Robosuite task name")
    parser.add_argument("--test-vlm-only", action="store_true",
                        help="Only test VLM client connection")
    parser.add_argument("--full-rollout", action="store_true",
                        help="Run full rollout test")
    parser.add_argument("--test-async", action="store_true",
                        help="Test async wrapper")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of test steps")
    args = parser.parse_args()

    try:
        # Always test VLM connection first
        test_vlm_client_connection(args.api_base)

        if args.test_vlm_only:
            return

        if args.full_rollout or not args.test_async:
            test_with_robosuite(args.task, args.api_base, args.num_steps)

        if args.test_async:
            test_async_wrapper(args.task, args.api_base, args.num_steps)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
