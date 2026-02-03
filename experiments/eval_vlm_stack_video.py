"""
Evaluate VLM on Stack task with video recording and skill selection bar.
"""
import os
import json
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load variant config from training
exp_dir = "/home/fxxk/git/LLM_LTL/data/vlm-maple-Stack/01-31-td-only-v1/01-31-td-only-v1_2026_01_31_00_53_44_0000--s-0"
with open(os.path.join(exp_dir, 'variant.json'), 'r') as f:
    variant = json.load(f)

# Load latest checkpoint
print("Loading policy...")
ckpt_path = os.path.join(exp_dir, "itr_300.pkl")
data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
policy = data['evaluation/policy']
print(f"Policy type: {type(policy).__name__}")
print(f"Checkpoint: {ckpt_path}")

# Create environment using exact training config
print("\nCreating environment...")
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

env_variant = variant['env_variant']
controller_config = load_controller_config(default_controller=env_variant['controller_type'])
controller_config_update = env_variant.get('controller_config_update', {})
controller_config.update(controller_config_update)

# Get env kwargs but override camera size for video
env_kwargs = env_variant['env_kwargs'].copy()
env_kwargs['camera_heights'] = 480
env_kwargs['camera_widths'] = 480

env = suite.make(
    env_name=env_variant['env_type'],
    robots='Panda',
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    controller_configs=controller_config,
    **env_kwargs
)

obs_keys = env_variant['robot_keys'] + env_variant['obj_keys']
env = GymWrapper(env, keys=obs_keys)

# Get skill names from config
skill_config = env_kwargs.get('skill_config', {})
SKILLS = skill_config.get('skills', ["atomic", "reach", "grasp", "push"])

# Create VLM client
print("\nInitializing VLM...")
from llm_ltl.vlm import QwenVLClient
from llm_ltl.vlm.vlm_reward_wrapper import VLMRewardWrapper

vlm_variant = variant.get('vlm_variant', {})
vlm_client = QwenVLClient(
    api_base=vlm_variant.get('api_base', 'http://172.19.1.40:8001'),
    model_name=vlm_variant.get('model_name', 'Qwen/Qwen3-VL-8B-Instruct')
)

env = VLMRewardWrapper(
    env=env,
    task_description=vlm_variant.get('task_description', 'Stack the red cube on top of the green cube'),
    skill_controller=getattr(env.env, 'skill_controller', None),
    vlm_client=vlm_client,
    vlm_reward_scale=vlm_variant.get('vlm_reward_scale', 0.5),
    eval_frequency=5,  # More frequent for video evaluation
    binary_weight=vlm_variant.get('binary_weight', 0.5),
    progress_weight=vlm_variant.get('progress_weight', 0.5),
    camera_name=vlm_variant.get('camera_name', 'frontview'),
    camera_height=480,
    camera_width=480,
    reward_mode=vlm_variant.get('reward_mode', 'bonus_only'),
    warmup_steps=0,
    image_history_size=vlm_variant.get('image_history_size', 4),
)

print(f"Skills: {SKILLS}")
SKILL_COLORS = {
    "atomic": (100, 100, 255),   # Blue
    "reach": (100, 255, 100),    # Green
    "grasp": (255, 100, 100),    # Red
    "push": (255, 200, 100),     # Orange
}

def add_skill_bar(frame, skill_name, step, vlm_info=None):
    """Add skill selection bar and VLM info to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = font

    width, height = img.size
    bar_height = 60

    # Draw skill bar background
    draw.rectangle([0, 0, width, bar_height], fill=(40, 40, 40))

    # Draw skill boxes
    box_width = width // len(SKILLS)
    for i, skill in enumerate(SKILLS):
        x1 = i * box_width
        x2 = (i + 1) * box_width - 2

        # Highlight selected skill
        if skill == skill_name:
            color = SKILL_COLORS.get(skill, (200, 200, 200))
            draw.rectangle([x1, 5, x2, bar_height - 5], fill=color, outline=(255, 255, 255), width=2)
            text_color = (0, 0, 0)
        else:
            draw.rectangle([x1, 5, x2, bar_height - 5], fill=(80, 80, 80), outline=(120, 120, 120))
            text_color = (150, 150, 150)

        # Draw skill name
        text_bbox = draw.textbbox((0, 0), skill.upper(), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x1 + (box_width - text_width) // 2
        draw.text((text_x, 15), skill.upper(), fill=text_color, font=font)

    # Draw step counter
    step_text = f"Step: {step}"
    draw.text((10, height - 30), step_text, fill=(255, 255, 255), font=font_small)

    # Draw VLM info if available
    if vlm_info:
        vlm_text = f"VLM: {'✓' if vlm_info['reasonable'] else '✗'} | Conf: {vlm_info['confidence']:.0%} | Prog: {vlm_info['progress']:.0%}"
        # Background for VLM info
        draw.rectangle([0, height - 35, width, height], fill=(40, 40, 40, 200))
        draw.text((10, height - 30), vlm_text, fill=(255, 255, 100) if vlm_info['reasonable'] else (255, 100, 100), font=font_small)

    return np.array(img)

# Output directory
output_dir = '/home/fxxk/git/LLM_LTL/data/stack_vlm_videos'
os.makedirs(output_dir, exist_ok=True)

# Run multiple episodes
print("\n" + "="*60)
print("Running 5 episodes with VLM evaluation")
print("="*60 + "\n")

max_steps = 200
num_episodes = 5
all_results = []

for episode in range(1, num_episodes + 1):
    obs = env.reset()
    frames = []
    vlm_data = []
    current_vlm_info = None

    print(f"--- Episode {episode}/{num_episodes} ---")

    for step in range(max_steps):
        with torch.no_grad():
            action, _ = policy.get_action(obs)

        # Get skill name
        skill_idx = int(action[0]) if len(action) > 0 else 0
        skill_name = SKILLS[skill_idx] if 0 <= skill_idx < len(SKILLS) else "?"

        # Capture frame with skill bar
        frame = env.env.env.sim.render(
            camera_name='frontview',
            width=480,
            height=480,
            depth=False
        )
        frame = np.flipud(frame)
        frame_with_bar = add_skill_bar(frame, skill_name, step, current_vlm_info)
        frames.append(frame_with_bar)

        obs, reward, done, info = env.step(action)

        # Record VLM evaluation
        if info.get('vlm_binary_confidence', 0) > 0:
            current_vlm_info = {
                'reasonable': info.get('vlm_binary_reasonable', 0),
                'confidence': info.get('vlm_binary_confidence', 0),
                'progress': info.get('vlm_progress', 0),
            }

            vlm_data.append({
                'step': step,
                'skill': skill_name,
                'reasonable': current_vlm_info['reasonable'],
                'confidence': current_vlm_info['confidence'],
                'progress': current_vlm_info['progress'],
            })

            print(f"Step {step}: skill={skill_name}, reasonable={current_vlm_info['reasonable']}, "
                  f"conf={current_vlm_info['confidence']:.2f}, prog={current_vlm_info['progress']:.2f}")

        if done:
            break

    success = info.get('success', False)
    print(f"\nEpisode {episode} finished: success={success}, steps={step+1}")

    all_results.append({
        'episode': episode,
        'success': success,
        'steps': step + 1,
        'vlm_data': vlm_data.copy(),
        'frames': frames.copy()
    })

# Save all videos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n{'='*60}")
print("Saving videos...")

for result in all_results:
    ep = result['episode']
    status = "success" if result['success'] else "fail"
    video_path = os.path.join(output_dir, f'stack_ep{ep}_{timestamp}_{status}.mp4')
    imageio.mimsave(video_path, result['frames'], fps=30)
    print(f"  Episode {ep}: {video_path} ({len(result['frames'])} frames)")

# Use last episode's data for main variables
success = all_results[-1]['success']
vlm_data = all_results[-1]['vlm_data']
frames = all_results[-1]['frames']

# Summary
success_count = sum(1 for r in all_results if r['success'])
print(f"\nSuccess rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.0f}%)")

# Generate VLM plot for all episodes
import pandas as pd

# Combine all VLM data
all_vlm_data = []
for result in all_results:
    for d in result['vlm_data']:
        d['episode'] = result['episode']
        d['success'] = result['success']
        all_vlm_data.append(d)

if all_vlm_data:
    df = pd.DataFrame(all_vlm_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'VLM Evaluation on Stack Task (td-only itr_300) - {success_count}/{num_episodes} Success',
                 fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot 1: Reasonable per episode
    ax1 = axes[0, 0]
    for i, ep in enumerate(df['episode'].unique()):
        ep_data = df[df['episode'] == ep]
        label = f"Ep{ep} ({'✓' if ep_data['success'].iloc[0] else '✗'})"
        ax1.plot(ep_data['step'], ep_data['reasonable'], marker='o',
                 color=colors[i % len(colors)], linewidth=2, alpha=0.7, label=label)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reasonable (0/1)')
    ax1.set_title('VLM Reasonable Assessment')
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confidence per episode
    ax2 = axes[0, 1]
    for i, ep in enumerate(df['episode'].unique()):
        ep_data = df[df['episode'] == ep]
        label = f"Ep{ep} ({'✓' if ep_data['success'].iloc[0] else '✗'})"
        ax2.plot(ep_data['step'], ep_data['confidence'], marker='s',
                 color=colors[i % len(colors)], linewidth=2, alpha=0.7, label=label)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Confidence')
    ax2.set_title('VLM Confidence')
    ax2.set_ylim(0.5, 1.0)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Progress per episode
    ax3 = axes[1, 0]
    for i, ep in enumerate(df['episode'].unique()):
        ep_data = df[df['episode'] == ep]
        label = f"Ep{ep} ({'✓' if ep_data['success'].iloc[0] else '✗'})"
        ax3.plot(ep_data['step'], ep_data['progress'], marker='^',
                 color=colors[i % len(colors)], linewidth=2, alpha=0.7, label=label)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Progress')
    ax3.set_title('VLM Progress Assessment')
    ax3.set_ylim(-0.1, 1.0)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Skill distribution
    ax4 = axes[1, 1]
    skill_counts = df['skill'].value_counts()
    bar_colors = [tuple(c/255 for c in SKILL_COLORS.get(s, (128, 128, 128))) for s in skill_counts.index]
    ax4.bar(skill_counts.index, skill_counts.values, color=bar_colors)
    ax4.set_xlabel('Skill')
    ax4.set_ylabel('Count')
    ax4.set_title('Skill Usage Distribution (all episodes)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'vlm_eval_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

env.close()
print("\n" + "="*60)
print("Done!")
print(f"Video: {video_path}")
if vlm_data:
    print(f"Plot: {plot_path}")
print("="*60)
