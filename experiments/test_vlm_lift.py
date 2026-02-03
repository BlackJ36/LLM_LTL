"""
Simple test script for Lift policy with VLM evaluation.
"""
import os
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load policy (torch format)
print("Loading policy...")
ckpt_path = "/home/fxxk/git/LLM_LTL/data/lift/01-19-test/01-19-test_2026_01_19_17_07_28_0000--s-45612/itr_150.pkl"
data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
policy = data['evaluation/policy']
print(f"Policy type: {type(policy).__name__}")

# Create environment
print("\nCreating environment...")
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

controller_config = load_controller_config(default_controller='OSC_POSITION_YAW')
controller_config['position_limits'] = [[-0.3, -0.3, 0.75], [0.15, 0.3, 1.15]]

# Skill config from trained Lift model
skill_config = {
    "aff_penalty_fac": 15.0,
    "skills": ["atomic", "open", "reach", "grasp", "push"],
    "base_config": {
        "global_xyz_bounds": [[-0.3, -0.3, 0.8], [0.15, 0.3, 0.95]],
        "lift_height": 0.95,
        "binary_gripper": True,
        "aff_threshold": 0.06,
        "aff_type": "dense",
        "aff_tanh_scaling": 10.0
    },
    "atomic_config": {"use_ori_params": True},
    "reach_config": {
        "local_xyz_scale": [0.0, 0.0, 0.06],
        "max_ac_calls": 15,
        "use_gripper_params": False,
        "use_ori_params": False
    },
    "grasp_config": {
        "aff_threshold": 0.03,
        "global_xyz_bounds": [[-0.3, -0.3, 0.8], [0.15, 0.3, 0.85]],
        "local_xyz_scale": [0.0, 0.0, 0.0],
        "max_ac_calls": 20,
        "num_grasp_steps": 3,
        "num_reach_steps": 2,
        "use_ori_params": True
    },
    "push_config": {
        "aff_threshold": [0.12, 0.12, 0.04],
        "delta_xyz_scale": [0.25, 0.25, 0.05],
        "global_xyz_bounds": [[-0.3, -0.3, 0.8], [0.15, 0.3, 0.85]],
        "max_ac_calls": 20,
        "use_ori_params": True
    },
}

env = suite.make(
    env_name='Lift',
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    controller_configs=controller_config,
    ignore_done=True,
    reward_shaping=True,
    hard_reset=False,
    control_freq=10,
    camera_heights=256,
    camera_widths=256,
    skill_config=skill_config,
)

robot_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
obj_keys = ['object-state']
env = GymWrapper(env, keys=robot_keys + obj_keys)

# Create VLM client
print("\nInitializing VLM...")
from llm_ltl.vlm import QwenVLClient
from llm_ltl.vlm.vlm_reward_wrapper import VLMRewardWrapper

vlm_client = QwenVLClient(
    api_base='http://172.19.1.40:8001',
    model_name='Qwen/Qwen3-VL-8B-Instruct'
)

# Wrap with VLM (eval every 5 steps)
env = VLMRewardWrapper(
    env=env,
    task_description='Lift the cube off the table',
    skill_controller=getattr(env.env, 'skill_controller', None),
    vlm_client=vlm_client,
    vlm_reward_scale=0.5,
    eval_frequency=5,
    binary_weight=0.5,
    progress_weight=0.5,
    camera_name='frontview',
    camera_height=256,
    camera_width=256,
    reward_mode='bonus_only',
    warmup_steps=0,
    image_history_size=4,
)

# Run episodes
print("\n" + "="*60)
print("Running episodes with VLM evaluation (every 5 steps)")
print("="*60 + "\n")

num_episodes = 5
max_steps = 150
results = []

for ep in range(num_episodes):
    obs = env.reset()
    ep_return = 0
    ep_vlm_return = 0
    vlm_reasonable = []
    vlm_confidence = []
    vlm_progress = []

    for step in range(max_steps):
        with torch.no_grad():
            action, _ = policy.get_action(obs)

        obs, reward, done, info = env.step(action)
        ep_return += reward
        ep_vlm_return += info.get('vlm_reward_scaled', 0)

        # Collect VLM metrics on every step (filter non-zero later)
        conf = info.get('vlm_binary_confidence', 0)
        if conf > 0:  # Only collect when VLM actually evaluated
            vlm_reasonable.append(info.get('vlm_binary_reasonable', 0))
            vlm_confidence.append(conf)
            vlm_progress.append(info.get('vlm_progress', 0))
            reasonable = info.get('vlm_binary_reasonable')
            prog = info.get('vlm_progress')
            # Get primitive name from action (skill index is first element)
            skill_idx = int(action[0]) if len(action) > 0 else -1
            skills = ["atomic", "open", "reach", "grasp", "push"]
            skill_name = skills[skill_idx] if 0 <= skill_idx < len(skills) else "?"
            print(f"  Step {step}: skill={skill_name}, reasonable={reasonable}, conf={conf:.2f}, prog={prog:.2f}")

        if done:
            break

    success = info.get('success', False)
    reasonable_rate = np.mean(vlm_reasonable) if vlm_reasonable else 0
    avg_confidence = np.mean(vlm_confidence) if vlm_confidence else 0
    final_progress = vlm_progress[-1] if vlm_progress else 0

    results.append({
        'return': ep_return,
        'vlm_return': ep_vlm_return,
        'success': success,
        'reasonable_rate': reasonable_rate,
        'avg_confidence': avg_confidence,
        'final_progress': final_progress,
    })

    print(f"\nEpisode {ep+1}/{num_episodes}:")
    print(f"  Return: {ep_return:.2f}")
    print(f"  VLM Return: {ep_vlm_return:.2f}")
    print(f"  Success: {success}")
    print(f"  VLM Reasonable Rate: {reasonable_rate:.2%}")
    print(f"  VLM Avg Confidence: {avg_confidence:.2%}")
    print(f"  VLM Final Progress: {final_progress:.2%}")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Success Rate: {np.mean([r['success'] for r in results]):.2%}")
print(f"Mean Return: {np.mean([r['return'] for r in results]):.2f}")
print(f"Mean VLM Return: {np.mean([r['vlm_return'] for r in results]):.2f}")
print(f"Mean VLM Reasonable Rate: {np.mean([r['reasonable_rate'] for r in results]):.2%}")
print(f"Mean VLM Confidence: {np.mean([r['avg_confidence'] for r in results]):.2%}")
print(f"Mean VLM Final Progress: {np.mean([r['final_progress'] for r in results]):.2%}")

env.close()
