"""
Full evaluation of Stack model with:
1. 50 episodes evaluation
2. Videos with VLM decisions and skill bar
3. Skillmap visualization (primitive selection across all episodes)
"""
import os
import sys
import json
import numpy as np
import torch
import imageio
import matplotlib

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colorbar
import math
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load variant config from training
exp_dir = "/home/fxxk/git/LLM_LTL/data/vlm-maple-Stack/01-31-td-only-v1/01-31-td-only-v1_2026_01_31_00_53_44_0000--s-0"
with open(os.path.join(exp_dir, 'variant.json'), 'r') as f:
    variant = json.load(f)

# Use latest checkpoint (itr_500 or params.pkl)
ckpt_files = ['itr_500.pkl', 'itr_475.pkl', 'itr_450.pkl', 'params.pkl']
ckpt_path = None
for cf in ckpt_files:
    cp = os.path.join(exp_dir, cf)
    if os.path.exists(cp):
        ckpt_path = cp
        break

print(f"Loading policy from: {ckpt_path}")
data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
policy = data['evaluation/policy']
print(f"Policy type: {type(policy).__name__}")

# Create environment
print("\nCreating environment...")
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

env_variant = variant['env_variant']
controller_config = load_controller_config(default_controller=env_variant['controller_type'])
controller_config_update = env_variant.get('controller_config_update', {})
controller_config.update(controller_config_update)

env_kwargs = env_variant['env_kwargs'].copy()
env_kwargs['camera_heights'] = 256
env_kwargs['camera_widths'] = 256

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

# Get skill info
skill_config = env_kwargs.get('skill_config', {})
SKILLS = skill_config.get('skills', ["atomic", "reach", "grasp", "push"])
SKILL_COLORS = {
    "atomic": (100, 100, 255),
    "reach": (100, 255, 100),
    "grasp": (255, 100, 100),
    "push": (255, 200, 100),
}
SKILL_COLORS_MPL = {
    "atomic": "blue",
    "reach": "green",
    "grasp": "red",
    "push": "orange",
}

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
    eval_frequency=10,  # Every 10 steps for faster eval
    binary_weight=vlm_variant.get('binary_weight', 0.5),
    progress_weight=vlm_variant.get('progress_weight', 0.5),
    camera_name=vlm_variant.get('camera_name', 'frontview'),
    camera_height=256,
    camera_width=256,
    reward_mode=vlm_variant.get('reward_mode', 'bonus_only'),
    warmup_steps=0,
    image_history_size=vlm_variant.get('image_history_size', 4),
)

print(f"Skills: {SKILLS}")

def add_skill_bar(frame, skill_name, step, vlm_info=None, success=False):
    """Add skill selection bar and VLM info to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = font

    width, height = img.size
    bar_height = 40

    # Draw skill bar background
    draw.rectangle([0, 0, width, bar_height], fill=(40, 40, 40))

    # Draw skill boxes
    box_width = width // len(SKILLS)
    for i, skill in enumerate(SKILLS):
        x1 = i * box_width
        x2 = (i + 1) * box_width - 2

        if skill == skill_name:
            color = SKILL_COLORS.get(skill, (200, 200, 200))
            draw.rectangle([x1, 3, x2, bar_height - 3], fill=color, outline=(255, 255, 255), width=2)
            text_color = (0, 0, 0)
        else:
            draw.rectangle([x1, 3, x2, bar_height - 3], fill=(80, 80, 80), outline=(120, 120, 120))
            text_color = (150, 150, 150)

        text_bbox = draw.textbbox((0, 0), skill.upper(), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x1 + (box_width - text_width) // 2
        draw.text((text_x, 10), skill.upper(), fill=text_color, font=font)

    # Draw step and success indicator
    step_text = f"Step: {step}"
    if success:
        step_text += " | SUCCESS"
        draw.rectangle([0, height - 25, width, height], fill=(0, 150, 0))
    draw.text((5, height - 22), step_text, fill=(255, 255, 255), font=font_small)

    # Draw VLM info
    if vlm_info:
        vlm_text = f"VLM: {'✓' if vlm_info['reasonable'] else '✗'} Prog:{vlm_info['progress']:.0%}"
        color = (255, 255, 100) if vlm_info['reasonable'] else (255, 100, 100)
        draw.text((width - 120, height - 22), vlm_text, fill=color, font=font_small)

    return np.array(img)

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'/home/fxxk/git/LLM_LTL/data/stack_eval_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Run evaluation
print("\n" + "="*60)
print("Running 50 episodes evaluation")
print("="*60 + "\n")

num_episodes = 50
max_steps = 200
all_results = []
all_skill_ids = []
all_vlm_data = []

for episode in range(1, num_episodes + 1):
    obs = env.reset()
    frames = []
    vlm_data = []
    skill_ids = []
    current_vlm_info = None
    success = False

    for step in range(max_steps):
        with torch.no_grad():
            action, _ = policy.get_action(obs)

        skill_idx = int(action[0]) if len(action) > 0 else 0
        skill_name = SKILLS[skill_idx] if 0 <= skill_idx < len(SKILLS) else "?"
        skill_ids.append(skill_idx)

        # Capture frame (only for first 5 episodes to save memory)
        if episode <= 5:
            frame = env.env.env.sim.render(camera_name='frontview', width=256, height=256, depth=False)
            frame = np.flipud(frame)
            frame_with_bar = add_skill_bar(frame, skill_name, step, current_vlm_info, success)
            frames.append(frame_with_bar)

        obs, reward, done, info = env.step(action)
        success = info.get('success', False)

        # Record VLM evaluation
        if info.get('vlm_binary_confidence', 0) > 0:
            current_vlm_info = {
                'reasonable': info.get('vlm_binary_reasonable', 0),
                'confidence': info.get('vlm_binary_confidence', 0),
                'progress': info.get('vlm_progress', 0),
            }
            vlm_data.append({
                'episode': episode,
                'step': step,
                'skill': skill_name,
                **current_vlm_info
            })

        if done:
            break

    # Pad skill_ids to max_steps
    while len(skill_ids) < max_steps:
        skill_ids.append(-1)
    all_skill_ids.append(skill_ids[:max_steps])

    result = {
        'episode': episode,
        'success': success,
        'steps': step + 1,
        'vlm_data': vlm_data,
    }
    all_results.append(result)
    all_vlm_data.extend(vlm_data)

    # Save video for first 5 episodes
    if episode <= 5 and frames:
        status = "success" if success else "fail"
        video_path = os.path.join(output_dir, f'ep{episode:02d}_{status}.mp4')
        imageio.mimsave(video_path, frames, fps=30)

    # Progress
    success_count = sum(1 for r in all_results if r['success'])
    print(f"Episode {episode:2d}/50: {'✓' if success else '✗'} | Success rate: {success_count}/{episode} ({success_count/episode*100:.1f}%)")

env.close()

# Final statistics
success_count = sum(1 for r in all_results if r['success'])
print(f"\n{'='*60}")
print(f"FINAL RESULTS: {success_count}/50 ({success_count/50*100:.1f}%) Success")
print(f"{'='*60}")

# Generate Skillmap
print("\nGenerating skillmap...")
skill_ids_array = np.array(all_skill_ids)
num_skills = len(SKILLS)
colors = ["white"] + [SKILL_COLORS_MPL[s] for s in SKILLS]

# Create faint colors for success
colors_rgba = [matplotlib.colors.to_rgba(c) for c in colors]
colors_faint_rgba = []
for i, rgba in enumerate(colors_rgba):
    if i > 0:
        rgba_faint = list(rgba)
        rgba_faint[-1] = 0.3
        colors_faint_rgba.append(tuple(rgba_faint))

cmap = ListedColormap(colors_rgba + colors_faint_rgba)
cmap_vis = ListedColormap(colors_rgba)

# Mark success episodes with faint colors
for i, result in enumerate(all_results):
    if result['success']:
        for j in range(max_steps):
            if skill_ids_array[i, j] >= 0:
                skill_ids_array[i, j] += num_skills

fig_width = 15 * math.ceil(max_steps / 100)
plt.figure(figsize=(fig_width, 12))
plt.pcolormesh(skill_ids_array, edgecolors='w', linewidth=0.5, cmap=cmap, vmin=-1.5, vmax=num_skills*2 - 0.5)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Episode', fontsize=12)
plt.title(f'Stack Task - Skill Selection Map (td-only, {success_count}/50 Success)', fontsize=14)

# Add colorbar
cax, _ = matplotlib.colorbar.make_axes(plt.gca(), location='right')
ticks = 1/(num_skills+1) * (np.arange(1, num_skills + 1) + 0.5)
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap_vis, ticks=ticks)
cbar.ax.set_yticklabels(SKILLS)

skillmap_path = os.path.join(output_dir, 'skillmap.png')
plt.savefig(skillmap_path, bbox_inches='tight', dpi=150)
print(f"Skillmap saved to {skillmap_path}")

# Generate VLM summary plot
if all_vlm_data:
    df = pd.DataFrame(all_vlm_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'VLM Evaluation Summary - Stack Task ({success_count}/50 Success)', fontsize=14, fontweight='bold')

    # Plot 1: Reasonable rate by episode
    ax1 = axes[0, 0]
    ep_reasonable = df.groupby('episode')['reasonable'].mean()
    colors = ['green' if all_results[i-1]['success'] else 'red' for i in ep_reasonable.index]
    ax1.bar(ep_reasonable.index, ep_reasonable.values, color=colors, alpha=0.7)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reasonable Rate')
    ax1.set_title('VLM Reasonable Rate by Episode')
    ax1.set_ylim(0, 1.1)

    # Plot 2: Progress over steps (averaged)
    ax2 = axes[0, 1]
    step_progress = df.groupby('step')['progress'].mean()
    ax2.plot(step_progress.index, step_progress.values, marker='o', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Average Progress')
    ax2.set_title('Average VLM Progress over Steps')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Skill usage
    ax3 = axes[1, 0]
    skill_counts = df['skill'].value_counts()
    bar_colors = [tuple(c/255 for c in SKILL_COLORS.get(s, (128, 128, 128))) for s in skill_counts.index]
    ax3.bar(skill_counts.index, skill_counts.values, color=bar_colors)
    ax3.set_xlabel('Skill')
    ax3.set_ylabel('Count')
    ax3.set_title('Skill Usage Distribution (at VLM eval steps)')

    # Plot 4: Success vs Fail comparison
    ax4 = axes[1, 1]
    success_eps = [r['episode'] for r in all_results if r['success']]
    fail_eps = [r['episode'] for r in all_results if not r['success']]

    success_reasonable = df[df['episode'].isin(success_eps)]['reasonable'].mean() if success_eps else 0
    fail_reasonable = df[df['episode'].isin(fail_eps)]['reasonable'].mean() if fail_eps else 0
    success_progress = df[df['episode'].isin(success_eps)]['progress'].mean() if success_eps else 0
    fail_progress = df[df['episode'].isin(fail_eps)]['progress'].mean() if fail_eps else 0

    x = np.arange(2)
    width = 0.35
    ax4.bar(x - width/2, [success_reasonable, success_progress], width, label='Success', color='green', alpha=0.7)
    ax4.bar(x + width/2, [fail_reasonable, fail_progress], width, label='Fail', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Reasonable Rate', 'Progress'])
    ax4.set_ylabel('Value')
    ax4.set_title('VLM Metrics: Success vs Fail Episodes')
    ax4.legend()
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    vlm_plot_path = os.path.join(output_dir, 'vlm_summary.png')
    plt.savefig(vlm_plot_path, dpi=150)
    print(f"VLM summary saved to {vlm_plot_path}")

# Save results to CSV
results_df = pd.DataFrame([{
    'episode': r['episode'],
    'success': r['success'],
    'steps': r['steps']
} for r in all_results])
results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

print(f"\n{'='*60}")
print(f"All outputs saved to: {output_dir}")
print(f"  - Videos: ep01-05_*.mp4")
print(f"  - Skillmap: skillmap.png")
print(f"  - VLM Summary: vlm_summary.png")
print(f"  - Results: results.csv")
print(f"{'='*60}")
