"""
Evaluate VLM on Door task and generate line charts.
Uses torch.load to load trained policy checkpoint.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load policy checkpoint (standard ML practice for loading trained models)
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
    camera_height=256,
    camera_width=256,
    reward_mode='bonus_only',
    warmup_steps=0,
    image_history_size=4,
)

# Run episodes and collect data
print("\n" + "="*60)
print("Running episodes with VLM evaluation")
print("="*60 + "\n")

num_episodes = 5
max_steps = 150
all_episode_data = []

for ep in range(num_episodes):
    obs = env.reset()
    episode_data = []

    for step in range(max_steps):
        with torch.no_grad():
            action, _ = policy.get_action(obs)

        obs, reward, done, info = env.step(action)

        # Record data at each step
        step_data = {
            'episode': ep,
            'step': step,
            'env_reward': reward - info.get('vlm_reward_scaled', 0),
            'vlm_reward': info.get('vlm_reward_scaled', 0),
            'total_reward': reward,
            'vlm_reasonable': info.get('vlm_binary_reasonable', np.nan),
            'vlm_confidence': info.get('vlm_binary_confidence', np.nan),
            'vlm_progress': info.get('vlm_progress', np.nan),
        }

        # Only record VLM metrics when VLM evaluated
        if info.get('vlm_binary_confidence', 0) > 0:
            skill_idx = int(action[0]) if len(action) > 0 else -1
            skills = ["atomic", "grasp", "reach_osc", "push", "open"]
            step_data['skill'] = skills[skill_idx] if 0 <= skill_idx < len(skills) else "?"
            print(f"Ep {ep+1} Step {step}: skill={step_data['skill']}, reasonable={step_data['vlm_reasonable']}, conf={step_data['vlm_confidence']:.2f}, prog={step_data['vlm_progress']:.2f}")
        else:
            step_data['skill'] = None

        episode_data.append(step_data)

        if done:
            break

    all_episode_data.extend(episode_data)
    success = info.get('success', False)
    print(f"\nEpisode {ep+1} finished: success={success}\n")

env.close()

# Convert to DataFrame
df = pd.DataFrame(all_episode_data)

# Save to CSV
csv_path = '/home/fxxk/git/LLM_LTL/data/door_vlm_eval_results.csv'
df.to_csv(csv_path, index=False)
print(f"\nData saved to {csv_path}")

# Filter VLM evaluation steps only
vlm_df = df[df['vlm_confidence'].notna()].copy()
vlm_df['vlm_step'] = vlm_df.groupby('episode').cumcount()

# Generate plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: VLM Reasonable over steps (per episode)
ax1 = axes[0, 0]
for ep in vlm_df['episode'].unique():
    ep_data = vlm_df[vlm_df['episode'] == ep]
    ax1.plot(ep_data['vlm_step'], ep_data['vlm_reasonable'], marker='o', label=f'Episode {ep+1}', alpha=0.7)
ax1.set_xlabel('VLM Evaluation Step')
ax1.set_ylabel('Reasonable (0/1)')
ax1.set_title('VLM Reasonable Assessment over Steps')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Plot 2: VLM Confidence over steps
ax2 = axes[0, 1]
for ep in vlm_df['episode'].unique():
    ep_data = vlm_df[vlm_df['episode'] == ep]
    ax2.plot(ep_data['vlm_step'], ep_data['vlm_confidence'], marker='s', label=f'Episode {ep+1}', alpha=0.7)
ax2.set_xlabel('VLM Evaluation Step')
ax2.set_ylabel('Confidence')
ax2.set_title('VLM Confidence over Steps')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: VLM Progress over steps
ax3 = axes[1, 0]
for ep in vlm_df['episode'].unique():
    ep_data = vlm_df[vlm_df['episode'] == ep]
    ax3.plot(ep_data['vlm_step'], ep_data['vlm_progress'], marker='^', label=f'Episode {ep+1}', alpha=0.7)
ax3.set_xlabel('VLM Evaluation Step')
ax3.set_ylabel('Progress')
ax3.set_title('VLM Progress Assessment over Steps')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Average metrics across episodes
ax4 = axes[1, 1]
avg_by_step = vlm_df.groupby('vlm_step').agg({
    'vlm_reasonable': 'mean',
    'vlm_confidence': 'mean',
    'vlm_progress': 'mean'
}).reset_index()

ax4.plot(avg_by_step['vlm_step'], avg_by_step['vlm_reasonable'], marker='o', label='Reasonable', linewidth=2)
ax4.plot(avg_by_step['vlm_step'], avg_by_step['vlm_confidence'], marker='s', label='Confidence', linewidth=2)
ax4.plot(avg_by_step['vlm_step'], avg_by_step['vlm_progress'], marker='^', label='Progress', linewidth=2)
ax4.set_xlabel('VLM Evaluation Step')
ax4.set_ylabel('Value')
ax4.set_title('Average VLM Metrics over Steps')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = '/home/fxxk/git/LLM_LTL/data/door_vlm_eval_plot.png'
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")

plt.show()
print("\nDone!")
