"""
Example script to run MAPLE with VLM reward guidance.

This demonstrates how to integrate VLM-based reward signals
with the MAPLE hierarchical RL training.

Usage:
    # Test VLM connection first
    python -m llm_ltl.vlm.test_vlm_reward --test-vlm-only

    # Run training with VLM reward
    uv run python experiments/run_vlm_maple.py --task Lift --num-envs 3

    # Run without VLM (baseline)
    uv run python experiments/run_vlm_maple.py --task Lift --no-vlm

    # Debug mode (quick test)
    uv run python experiments/run_vlm_maple.py --task Lift --debug

    # View TensorBoard
    tensorboard --logdir data/vlm-maple-Lift-vlm0.5/
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maple.launchers.launcher_util import run_experiment
from llm_ltl.vlm.vlm_maple_launcher import experiment


def get_base_variant():
    """Get base configuration matching MAPLE defaults."""
    return {
        'layer_size': 256,
        'replay_buffer_size': int(1E6),
        'rollout_fn_kwargs': {
            'terminals_all_false': True,
        },
        'algorithm_kwargs': {
            'num_epochs': 500,
            'num_expl_steps_per_train_loop': 3000,
            'num_eval_steps_per_epoch': 3000,
            'num_trains_per_train_loop': 1000,
            'min_num_steps_before_training': 30000,
            'max_path_length': 150,
            'batch_size': 1024,
            'eval_epoch_freq': 10,
        },
        'trainer_kwargs': {
            'discount': 0.99,
            'soft_target_tau': 1e-3,
            'target_update_period': 1,
            'policy_lr': 3e-5,
            'qf_lr': 3e-5,
            'reward_scale': 1,
            'use_automatic_entropy_tuning': True,
        },
        'll_sac_variant': {
            'high_init_ent': True,
        },
        'pamdp_variant': {
            'one_hot_s': True,
            'high_init_ent': True,
            'one_hot_factor': 0.50,
        },
        'env_variant': {
            'robot_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
            'obj_keys': ['object-state'],
            'controller_type': 'OSC_POSITION_YAW',
            'controller_config_update': {
                'position_limits': [
                    [-0.30, -0.30, 0.75],
                    [0.15, 0.30, 1.15]
                ],
            },
            'env_kwargs': {
                'ignore_done': True,
                'reward_shaping': True,
                'hard_reset': False,
                'control_freq': 10,
                'camera_heights': 256,
                'camera_widths': 256,
                'table_offset': [-0.075, 0, 0.8],
                'reward_scale': 5.0,

                'skill_config': {
                    'skills': ['atomic', 'reach', 'grasp', 'push'],
                    'aff_penalty_fac': 15.0,
                    'base_config': {
                        'global_xyz_bounds': [
                            [-0.30, -0.30, 0.80],
                            [0.15, 0.30, 0.95]
                        ],
                        'lift_height': 0.95,
                        'binary_gripper': True,
                        'aff_threshold': 0.06,
                        'aff_type': 'dense',
                        'aff_tanh_scaling': 10.0,
                    },
                    'atomic_config': {
                        'use_ori_params': True,
                    },
                    'reach_config': {
                        'use_gripper_params': False,
                        'local_xyz_scale': [0.0, 0.0, 0.06],
                        'use_ori_params': False,
                        'max_ac_calls': 15,
                    },
                    'grasp_config': {
                        'global_xyz_bounds': [
                            [-0.30, -0.30, 0.80],
                            [0.15, 0.30, 0.85]
                        ],
                        'aff_threshold': 0.03,
                        'local_xyz_scale': [0.0, 0.0, 0.0],
                        'use_ori_params': True,
                        'max_ac_calls': 20,
                        'num_reach_steps': 2,
                        'num_grasp_steps': 3,
                    },
                    'push_config': {
                        'global_xyz_bounds': [
                            [-0.30, -0.30, 0.80],
                            [0.15, 0.30, 0.85]
                        ],
                        'delta_xyz_scale': [0.25, 0.25, 0.05],
                        'max_ac_calls': 20,
                        'use_ori_params': True,
                        'aff_threshold': [0.12, 0.12, 0.04],
                    },
                    'gripper_config': {
                        'max_ac_calls': 5,
                    },
                },
            },
        },
        'use_tensorboard': True,
        'save_video': True,
        'dump_video_kwargs': {
            'save_video_period': 50,
            'rows': 1,
            'columns': 4,
        },
    }


def get_task_description(task: str) -> str:
    """Get natural language task description for VLM."""
    descriptions = {
        'Lift': 'Pick up the red cube from the table and lift it upward',
        'Stack': 'Stack the red cube on top of the green cube',
        'NutAssemblyRound': 'Pick up the nut and place it on the peg',
        'PickPlaceCan': 'Pick up the can and place it in the bin',
        'Door': 'Open the door by pushing or pulling the handle',
        'Wipe': 'Wipe the table surface with the cloth',
    }
    return descriptions.get(task, f'Complete the {task} manipulation task')


def get_task_env_config(task: str) -> dict:
    """Get task-specific environment configuration overrides."""
    configs = {
        'Lift': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
            'base_config': {
                'global_xyz_bounds': [[-0.30, -0.30, 0.80], [0.15, 0.30, 0.95]],
                'lift_height': 0.95,
                'binary_gripper': True,
                'aff_threshold': 0.06,
                'aff_type': 'dense',
                'aff_tanh_scaling': 10.0,
            },
            'grasp_config': {
                'global_xyz_bounds': [[-0.30, -0.30, 0.80], [0.15, 0.30, 0.85]],
                'aff_threshold': 0.03,
            },
            'push_config': {
                'global_xyz_bounds': [[-0.30, -0.30, 0.80], [0.15, 0.30, 0.85]],
            },
        },
        'Door': {
            'controller_type': 'OSC_POSITION',
            'position_limits': [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]],
            'table_offset': None,  # Door doesn't use table_offset
            'skills': ['atomic', 'grasp', 'reach_osc', 'push', 'open'],
            'base_config': {
                'global_xyz_bounds': [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]],
                'lift_height': 1.15,
                'binary_gripper': True,
                'aff_threshold': 0.06,
                'aff_type': 'dense',
                'aff_tanh_scaling': 10.0,
            },
            'grasp_config': {
                'global_xyz_bounds': [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]],
                'aff_threshold': 0.06,
            },
            'push_config': {
                'global_xyz_bounds': [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]],
            },
        },
    }
    return configs.get(task, configs['Lift'])


def process_variant(variant, args):
    """Process variant based on command line arguments."""
    # Set task
    variant['env_variant']['env_type'] = args.task

    # Apply task-specific configuration
    task_config = get_task_env_config(args.task)
    variant['env_variant']['controller_type'] = task_config['controller_type']
    variant['env_variant']['controller_config_update']['position_limits'] = task_config['position_limits']

    # Update skill_config
    skill_config = variant['env_variant']['env_kwargs']['skill_config']
    skill_config['skills'] = task_config['skills']
    skill_config['base_config'].update(task_config['base_config'])
    skill_config['grasp_config'].update(task_config['grasp_config'])
    skill_config['push_config'].update(task_config['push_config'])

    # Task-specific env_kwargs
    if task_config.get('table_offset') is not None:
        variant['env_variant']['env_kwargs']['table_offset'] = task_config['table_offset']
    elif 'table_offset' in variant['env_variant']['env_kwargs']:
        del variant['env_variant']['env_kwargs']['table_offset']

    # Vectorized training configuration
    variant['num_expl_envs'] = args.num_envs
    variant['num_eval_envs'] = 1
    variant['use_dummy_vec_env'] = args.dummy_vec_env

    # Override min_num_steps_before_training if specified
    if args.min_steps is not None:
        variant['algorithm_kwargs']['min_num_steps_before_training'] = args.min_steps

    # Override replay buffer size if specified
    if args.replay_buffer_size is not None:
        variant['replay_buffer_size'] = args.replay_buffer_size

    # VLM configuration
    # Default warmup: same as min_num_steps_before_training (skip initial random exploration)
    # NOTE: In vectorized envs, steps are distributed across envs, so divide by num_envs
    min_steps = variant['algorithm_kwargs']['min_num_steps_before_training']
    num_envs = args.num_envs
    if args.vlm_no_warmup:
        vlm_warmup = 0
    elif args.vlm_warmup is not None:
        vlm_warmup = args.vlm_warmup // num_envs  # Adjust for vectorized envs
    else:
        vlm_warmup = min_steps // num_envs  # Each env sees total_steps/num_envs

    variant['vlm_variant'] = {
        'enabled': not args.no_vlm,
        'task_description': get_task_description(args.task),
        'api_base': args.vlm_api,
        'model_name': args.vlm_model,
        'vlm_reward_scale': args.vlm_scale,
        'eval_frequency': args.vlm_freq,
        'use_async': True,
        # Plan C: Soft action scoring (0-1)
        'action_weight': args.vlm_action_weight,
        'progress_weight': args.vlm_progress_weight,
        'camera_name': 'frontview',
        'camera_height': 256,
        'camera_width': 256,
        'warmup_steps': vlm_warmup,
        'image_history_size': args.vlm_history,
        # Plan B: Potential-based shaping
        'use_potential_shaping': args.use_potential_shaping,
        'gamma': 0.99,
        # Plan D: Curriculum learning
        'curriculum_mode': args.curriculum_mode,
        'curriculum_start': args.curriculum_start,
        'curriculum_end': args.curriculum_end,
        # VLM Priority Sampling configuration
        'priority_sampling': {
            'enabled': args.vlm_priority,
            'mode': args.vlm_priority_mode,  # 'vlm_only', 'td_only', 'hybrid'
            'alpha': args.vlm_priority_alpha,
            'beta_start': 0.4,
            'beta_end': 1.0,
            'beta_anneal_steps': 100000,
            # Simplified VLM score composition (action_score + progress)
            'action_score_weight': args.vlm_action_weight,
            'progress_weight': args.vlm_progress_weight,
            # Hybrid mode weights
            'vlm_weight': args.vlm_priority_vlm_weight,
            'td_weight': args.vlm_priority_td_weight,
            # Stability parameters
            'max_is_weight': args.vlm_priority_max_is_weight,
            'max_td_error': args.vlm_priority_max_td_error,
            # Buffer filtering
            'min_progress_threshold': args.vlm_min_progress,
        },
        # Early stopping
        'early_stop': {
            'enabled': args.early_stop,
            'threshold': args.early_stop_threshold,
            'patience': args.early_stop_patience,
        },
    }

    # Debug mode
    if args.debug:
        variant['algorithm_kwargs']['num_epochs'] = 50
        variant['algorithm_kwargs']['batch_size'] = 64
        steps = 100
        variant['algorithm_kwargs']['max_path_length'] = steps
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = steps
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = steps
        variant['algorithm_kwargs']['min_num_steps_before_training'] = steps
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 50
        variant['algorithm_kwargs']['eval_epoch_freq'] = 10  # Eval every 10 epochs
        variant['replay_buffer_size'] = int(1E4)
        variant['dump_video_kwargs']['save_video_period'] = 1
        variant['dump_video_kwargs']['columns'] = 2
        # Update VLM warmup for debug mode (unless explicitly set)
        if args.vlm_warmup is None and not args.vlm_no_warmup:
            variant['vlm_variant']['warmup_steps'] = steps // args.num_envs

    # Custom epochs
    if args.epochs:
        variant['algorithm_kwargs']['num_epochs'] = args.epochs

    # No video
    if args.no_video:
        variant['save_video'] = False

    return variant


def main():
    parser = argparse.ArgumentParser(
        description='Run MAPLE with VLM reward guidance'
    )

    # Task configuration
    parser.add_argument('--task', type=str, default='Lift',
                        choices=['Lift', 'Stack', 'NutAssemblyRound', 'PickPlaceCan', 'Door'],
                        help='Robosuite task')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 500)')

    # VLM configuration
    parser.add_argument('--no-vlm', action='store_true',
                        help='Disable VLM reward (baseline)')
    parser.add_argument('--vlm-api', type=str,
                        default='http://172.19.1.40:8001',
                        help='VLM API base URL')
    parser.add_argument('--vlm-model', type=str,
                        default='Qwen/Qwen3-VL-8B-Instruct',
                        help='VLM model name')
    parser.add_argument('--vlm-scale', type=float, default=0.5,
                        help='VLM reward scale factor (0.3-1.0 recommended)')
    parser.add_argument('--vlm-freq', type=int, default=1,
                        help='VLM evaluation frequency (1 = every step)')
    parser.add_argument('--vlm-action-weight', type=float, default=0.2,
                        help='Weight for action score (lower: MAPLE handles action selection)')
    parser.add_argument('--vlm-progress-weight', type=float, default=0.8,
                        help='Weight for progress (higher: encourages effective trajectories)')
    parser.add_argument('--vlm-warmup', type=int, default=None,
                        help='Steps before enabling VLM (default: same as min_num_steps_before_training)')
    parser.add_argument('--vlm-no-warmup', action='store_true',
                        help='Disable VLM warmup (enable VLM from step 0)')
    parser.add_argument('--vlm-history', type=int, default=4,
                        help='Number of historical images for VLM (default: 4, total 5 with current)')
    # Plan B: Potential-based shaping
    parser.add_argument('--use-potential-shaping', action='store_true', default=True,
                        help='Use potential-based progress shaping (allows negative reward)')
    parser.add_argument('--no-potential-shaping', dest='use_potential_shaping', action='store_false',
                        help='Disable potential-based shaping (positive delta only)')
    # Plan D: Curriculum learning
    parser.add_argument('--curriculum-mode', type=str, default='none',
                        choices=['none', 'linear', 'step', 'cosine'],
                        help='Curriculum mode for VLM weight decay')
    parser.add_argument('--curriculum-start', type=float, default=1.0,
                        help='Initial VLM weight for curriculum')
    parser.add_argument('--curriculum-end', type=float, default=0.1,
                        help='Final VLM weight for curriculum')
    # Buffer filtering
    parser.add_argument('--vlm-min-progress', type=float, default=0.0,
                        help='Min progress to store sample in buffer (filter ineffective trajectories)')

    # Early stopping
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help='Enable early stopping (default: enabled)')
    parser.add_argument('--no-early-stop', dest='early_stop', action='store_false',
                        help='Disable early stopping')
    parser.add_argument('--early-stop-threshold', type=float, default=0.95,
                        help='Success rate threshold for early stopping (default: 0.95)')
    parser.add_argument('--early-stop-patience', type=int, default=3,
                        help='Consecutive evals above threshold to stop (default: 3)')

    # VLM Priority Sampling
    parser.add_argument('--vlm-priority', action='store_true',
                        help='Enable VLM-based prioritized experience replay')
    parser.add_argument('--vlm-priority-mode', type=str, default='hybrid',
                        choices=['vlm_only', 'td_only', 'hybrid'],
                        help='Priority mode: vlm_only, td_only, or hybrid (default: hybrid)')
    parser.add_argument('--vlm-priority-alpha', type=float, default=0.6,
                        help='Priority exponent (0=uniform, 1=full priority)')
    parser.add_argument('--vlm-priority-vlm-weight', type=float, default=0.5,
                        help='VLM weight in hybrid mode (default: 0.5)')
    parser.add_argument('--vlm-priority-td-weight', type=float, default=0.5,
                        help='TD-error weight in hybrid mode (default: 0.5)')
    parser.add_argument('--vlm-priority-max-is-weight', type=float, default=10.0,
                        help='Max IS weight to prevent extreme updates (default: 10.0)')
    parser.add_argument('--vlm-priority-max-td-error', type=float, default=10.0,
                        help='Max TD-error for priority calculation (default: 10.0)')

    # Parallelization
    parser.add_argument('--num-envs', type=int, default=3,
                        help='Number of parallel exploration environments')
    parser.add_argument('--dummy-vec-env', action='store_true',
                        help='Use DummyVecEnv (sequential) instead of SubprocVecEnv')

    # GPU configuration
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')

    # Experiment configuration
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--label', type=str, default=None,
                        help='Experiment label for logging')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with minimal steps')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--min-steps', type=int, default=None,
                        help='Min steps before training (default: 30000, reduces initial exploration)')
    parser.add_argument('--replay-buffer-size', type=int, default=None,
                        help='Replay buffer size (default: 1M, use 100k for faster experiments)')
    parser.add_argument('--snapshot-gap', type=int, default=25,
                        help='Epochs between snapshots')

    args = parser.parse_args()

    # Build variant
    variant = get_base_variant()
    variant = process_variant(variant, args)

    # Experiment naming
    if args.no_vlm:
        exp_prefix = args.label or f"baseline"
    else:
        exp_prefix = args.label or f"vlm{args.vlm_scale}"

    # Print configuration
    print("=" * 60)
    print("MAPLE with VLM Reward")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"VLM enabled: {not args.no_vlm}")
    if not args.no_vlm:
        print(f"  API: {args.vlm_api}")
        print(f"  Reward scale: {args.vlm_scale}")
        print(f"  Eval frequency: every {args.vlm_freq} steps")
        print(f"  Weights: action={args.vlm_action_weight}, progress={args.vlm_progress_weight}")
        warmup = variant['vlm_variant']['warmup_steps']
        print(f"  Warmup steps: {warmup} (VLM enabled after {warmup} steps)")
        print(f"  Image history: {args.vlm_history} + 1 current = {args.vlm_history + 1} frames")
        print(f"  Task description: {get_task_description(args.task)}")
        print(f"  Plan B - Potential shaping: {args.use_potential_shaping}")
        print(f"  Plan D - Curriculum: {args.curriculum_mode} ({args.curriculum_start} -> {args.curriculum_end})")
        if args.vlm_priority:
            print(f"  Priority sampling: ENABLED")
            print(f"    Mode: {args.vlm_priority_mode}")
            print(f"    Alpha: {args.vlm_priority_alpha}")
            if args.vlm_priority_mode == 'hybrid':
                print(f"    Hybrid: VLM={args.vlm_priority_vlm_weight}, TD={args.vlm_priority_td_weight}")
            print(f"    Buffer filter: min_progress={args.vlm_min_progress}")
            print(f"    Stability: max_is_weight={args.vlm_priority_max_is_weight}, "
                  f"max_td_error={args.vlm_priority_max_td_error}")
    print(f"\nParallel envs: {args.num_envs}")
    print(f"Min steps before training: {variant['algorithm_kwargs']['min_num_steps_before_training']}")
    print(f"Debug mode: {args.debug}")
    print(f"GPU: {'disabled' if args.no_gpu else args.gpu}")
    print("=" * 60)
    print()

    # Run experiment using MAPLE's launcher
    run_experiment(
        experiment,
        exp_folder=f"vlm-maple-{args.task}",
        exp_prefix=exp_prefix,
        variant=variant,
        snapshot_mode='gap_and_last',
        snapshot_gap=args.snapshot_gap,
        seed=args.seed,
        use_gpu=(not args.no_gpu),
        gpu_id=args.gpu,
        mode='local',
        num_exps_per_instance=1,
    )


if __name__ == '__main__':
    main()
