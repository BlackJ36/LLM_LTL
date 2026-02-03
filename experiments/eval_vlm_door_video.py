"""
Evaluate VLM on Door task with video recording.
"""
import os
import numpy as np
import torch
import imageio
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load policy
print("Loading policy...")
ckpt_path = "/home/fxxk/git/LLM_LTL/data/vlm-maple-Door/01-30-vlm0.5/01-30-vlm0.5_2026_01_30_05_46_00_0000--s-0/itr_200.pkl"
data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
policy = data['evaluation/policy']
print(f"Policy type: {type(policy).__name__}")

# Create environment
print("\nCreating environment...")
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

controller_config = load_controller_config(default_controller='OSC_POSITION')

skill_config = {
    "aff_penalty_fac": 15.0,
    "skills": ["atomic", "grasp", "reach_osc", "push", "open"],
    "base_config": {
        "global_xyz_bounds": [[-0.25, -0.25, 0.9], [0.05, 0.05, 1.2]],
        "lift_height": 1.15,
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
        "aff_threshold": 0.06,
        "global_xyz_bounds": [[-0.25, -0.25, 0.9], [0.05, 0.05, 1.2]],
        "local_xyz_scale": [0.0, 0.0, 0.0],
        "max_ac_calls": 20,
        "num_grasp_steps": 3,
        "num_reach_steps": 2,
        "use_ori_params": True
    },
    "push_config": {
        "aff_threshold": [0.12, 0.12, 0.04],
        "delta_xyz_scale": [0.25, 0.25, 0.05],
        "global_xyz_bounds": [[-0.25, -0.25, 0.9], [0.05, 0.05, 1.2]],
        "max_ac_calls": 20,
        "use_ori_params": True
    },
    "gripper_config": {"max_ac_calls": 5}
}

env = suite.make(
    env_name='Door',
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    controller_configs=controller_config,
    ignore_done=True,
    reward_shaping=True,
    hard_reset=False,
    control_freq=10,
    camera_heights=480,
    camera_widths=480,
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

env = VLMRewardWrapper(
    env=env,
    task_description='Open the door by pushing the handle',
    skill_controller=getattr(env.env, 'skill_controller', None),
    vlm_client=vlm_client,
    vlm_reward_scale=0.5,
    eval_frequency=5,
    binary_weight=0.5,
    progress_weight=0.5,
    camera_name='frontview',
    camera_height=480,
    camera_width=480,
    reward_mode='bonus_only',
    warmup_steps=0,
    image_history_size=4,
)

# Output directory
output_dir = '/home/fxxk/git/LLM_LTL/data/door_vlm_videos'
os.makedirs(output_dir, exist_ok=True)

# Run episodes with video recording
print("\n" + "="*60)
print("Running episodes with VLM evaluation and video recording")
print("="*60 + "\n")

num_episodes = 2
max_steps = 150

for ep in range(num_episodes):
    obs = env.reset()
    frames = []
    vlm_data = []

    print(f"\n--- Episode {ep+1} ---")

    for step in range(max_steps):
        # Capture frame
        frame = env.env.env.sim.render(
            camera_name='frontview',
            width=480,
            height=480,
            depth=False
        )
        frame = np.flipud(frame)
        frames.append(frame)

        with torch.no_grad():
            action, _ = policy.get_action(obs)

        obs, reward, done, info = env.step(action)

        # Record VLM evaluation
        if info.get('vlm_binary_confidence', 0) > 0:
            skill_idx = int(action[0]) if len(action) > 0 else -1
            skills = ["atomic", "grasp", "reach_osc", "push", "open"]
            skill_name = skills[skill_idx] if 0 <= skill_idx < len(skills) else "?"

            vlm_data.append({
                'step': step,
                'skill': skill_name,
                'reasonable': info.get('vlm_binary_reasonable', 0),
                'confidence': info.get('vlm_binary_confidence', 0),
                'progress': info.get('vlm_progress', 0),
            })

            print(f"Step {step}: skill={skill_name}, reasonable={info.get('vlm_binary_reasonable')}, "
                  f"conf={info.get('vlm_binary_confidence'):.2f}, prog={info.get('vlm_progress'):.2f}")

        if done:
            break

    success = info.get('success', False)
    print(f"\nEpisode {ep+1} finished: success={success}, steps={step+1}")

    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f'door_ep{ep+1}_{timestamp}_{"success" if success else "fail"}.mp4')

    print(f"Saving video to {video_path}...")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved! ({len(frames)} frames)")

env.close()
print("\n" + "="*60)
print("Done!")
print(f"Videos saved to: {output_dir}")
print("="*60)
