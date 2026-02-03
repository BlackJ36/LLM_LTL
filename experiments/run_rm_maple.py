"""
Run MAPLE with LTL Reward Machine guidance.

This integrates formal LTL-based reward specifications with MAPLE training.

Usage:
    # Run with RM reward (Stack task)
    uv run python experiments/run_rm_maple.py --task stack --rm-weight 1.0

    # Run with RM reward only (replace env reward)
    uv run python experiments/run_rm_maple.py --task lift --rm-only

    # Run baseline (no RM)
    uv run python experiments/run_rm_maple.py --task stack --no-rm

    # Debug mode
    uv run python experiments/run_rm_maple.py --task stack --debug

    # View TensorBoard
    tensorboard --logdir data/rm-maple-stack/
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maple.launchers.launcher_util import run_experiment
from llm_ltl.reward_machines import create_rm, get_events_fn, get_available_tasks
from llm_ltl.reward_machines.rm_factory import get_available_reward_modes, print_rm_summary


def get_base_variant():
    """Get base configuration matching MAPLE defaults."""
    return {
        'layer_size': 256,
        'replay_buffer_size': int(1E5),  # 100k for faster experiments
        'rollout_fn_kwargs': {
            'terminals_all_false': True,
        },
        'algorithm_kwargs': {
            'num_epochs': 500,
            'num_expl_steps_per_train_loop': 3000,
            'num_eval_steps_per_epoch': 3000,
            'num_trains_per_train_loop': 500,  # Reduced: fewer CPU sampling ops
            'min_num_steps_before_training': 10000,
            'max_path_length': 150,
            'batch_size': 2048,  # Increased: better GPU utilization
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
        'save_video_period': 50,  # Separate from dump_video_kwargs
        'dump_video_kwargs': {
            'rows': 1,
            'columns': 4,
        },
    }


# Map task names to robosuite env types
TASK_ENV_MAP = {
    'lift': 'Lift',
    'door': 'Door',
    'pnp': 'PickPlaceCan',
    'pick_place': 'PickPlaceCan',
    'wipe': 'Wipe',
    'stack': 'Stack',
    'nut': 'NutAssemblyRound',
    'nut_assembly': 'NutAssemblyRound',
    'cleanup': 'Cleanup',
    'peg_ins': 'PegInHole',
    'peg_in_hole': 'PegInHole',
}


def get_task_env_type(task: str) -> str:
    """Get robosuite environment type for task."""
    task_lower = task.lower()
    if task_lower in TASK_ENV_MAP:
        return TASK_ENV_MAP[task_lower]
    # If not found, assume it's already a valid env type
    return task


def get_task_env_config(task: str) -> dict:
    """Get task-specific environment configuration."""
    task_lower = task.lower()

    configs = {
        'lift': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
        'stack': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
        'door': {
            'controller_type': 'OSC_POSITION',
            'position_limits': [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]],
            'table_offset': None,
            'skills': ['atomic', 'grasp', 'reach_osc', 'push', 'open'],
        },
        'pnp': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
        'wipe': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
        'nut': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
        'cleanup': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
        'peg_ins': {
            'controller_type': 'OSC_POSITION_YAW',
            'position_limits': [[-0.30, -0.30, 0.75], [0.15, 0.30, 1.15]],
            'table_offset': [-0.075, 0, 0.8],
            'skills': ['atomic', 'reach', 'grasp', 'push'],
        },
    }

    return configs.get(task_lower, configs['lift'])


def process_variant(variant, args):
    """Process variant based on command line arguments."""
    # Set task environment type
    env_type = get_task_env_type(args.task)
    variant['env_variant']['env_type'] = env_type

    # Apply task-specific configuration
    task_config = get_task_env_config(args.task)
    variant['env_variant']['controller_type'] = task_config['controller_type']
    variant['env_variant']['controller_config_update']['position_limits'] = task_config['position_limits']

    # Update skills
    skill_config = variant['env_variant']['env_kwargs']['skill_config']
    skill_config['skills'] = task_config['skills']

    # Task-specific table_offset
    if task_config.get('table_offset') is not None:
        variant['env_variant']['env_kwargs']['table_offset'] = task_config['table_offset']
    elif 'table_offset' in variant['env_variant']['env_kwargs']:
        del variant['env_variant']['env_kwargs']['table_offset']

    # Vectorized training
    variant['num_expl_envs'] = args.num_envs
    variant['num_eval_envs'] = 1
    variant['use_dummy_vec_env'] = args.dummy_vec_env

    # Override min_num_steps_before_training
    if args.min_steps is not None:
        variant['algorithm_kwargs']['min_num_steps_before_training'] = args.min_steps

    # Override replay buffer size
    if args.replay_buffer_size is not None:
        variant['replay_buffer_size'] = args.replay_buffer_size

    # RM configuration (based on IJCAI 2019 "LTL and Beyond" & TRAPs)
    variant['rm_variant'] = {
        'enabled': not args.no_rm,
        'task_name': args.task.lower(),
        'rm_reward_scale': args.rm_weight,
        'use_rm_only': args.rm_only,
        'extend_obs': args.rm_extend_obs,
        # Reward shaping parameters
        'reward_mode': args.rm_mode,
        'gamma': 0.99,
        'use_potential_shaping': args.rm_potential,
        'potential_scale': args.rm_potential_scale,
        'terminal_reward': args.rm_terminal_reward,
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
        variant['algorithm_kwargs']['eval_epoch_freq'] = 10
        variant['replay_buffer_size'] = int(1E4)
        variant['save_video_period'] = 1  # Use top-level key, not inside dump_video_kwargs
        variant['dump_video_kwargs']['columns'] = 2

    # Custom epochs
    if args.epochs:
        variant['algorithm_kwargs']['num_epochs'] = args.epochs

    # No video
    if args.no_video:
        variant['save_video'] = False

    return variant


def rm_experiment(variant):
    """
    Run MAPLE experiment with Reward Machine.

    This is the experiment function called by run_experiment.
    It wraps the standard MAPLE launcher with RM integration.
    """
    import torch
    from maple.launchers.robosuite_launcher import experiment as maple_experiment

    rm_variant = variant.get('rm_variant', {})

    if not rm_variant.get('enabled', True):
        # No RM, run standard MAPLE
        return maple_experiment(variant)

    # Get RM configuration
    task_name = rm_variant.get('task_name', 'stack')
    rm_reward_scale = rm_variant.get('rm_reward_scale', 1.0)
    use_rm_only = rm_variant.get('use_rm_only', False)
    extend_obs = rm_variant.get('extend_obs', False)

    # Reward shaping parameters
    reward_mode = rm_variant.get('reward_mode', 'hybrid')
    gamma = rm_variant.get('gamma', 0.99)
    use_potential_shaping = rm_variant.get('use_potential_shaping', True)
    potential_scale = rm_variant.get('potential_scale', 0.1)
    terminal_reward = rm_variant.get('terminal_reward', 1.0)

    # Create RM and events function
    rm = create_rm(
        task_name,
        reward_mode=reward_mode,
        gamma=gamma,
        use_potential_shaping=use_potential_shaping,
        potential_scale=potential_scale,
        terminal_reward=terminal_reward,
    )
    events_fn = get_events_fn(task_name)

    print(f"\n{'='*60}")
    print(f"Reward Machine Configuration")
    print(f"{'='*60}")
    print(f"Task: {task_name}")
    print(f"RM: {rm}")
    print(f"States: {sorted(rm.states)}")
    print(f"Terminal: {rm.terminal_states}")
    print(f"Reward scale: {rm_reward_scale}")
    print(f"RM only: {use_rm_only}")
    print(f"Extend obs: {extend_obs}")
    print(f"\nReward Shaping (IJCAI 2019 'LTL and Beyond'):")
    print(f"  Mode: {reward_mode}")
    print(f"  Potential shaping: {use_potential_shaping}")
    print(f"  Potential scale: {potential_scale}")
    print(f"  Terminal reward: {terminal_reward}")
    print(f"\nDistances to terminal:")
    for state in sorted(rm.states):
        dist = rm.distance_to_terminal(state)
        pot = rm.potential(state)
        print(f"  {state}: dist={dist}, Φ={pot:.2f}")
    print(f"{'='*60}\n")

    # Store RM in variant for use in launcher
    variant['_rm_instance'] = rm
    variant['_rm_events_fn'] = events_fn
    variant['_rm_reward_scale'] = rm_reward_scale
    variant['_rm_use_only'] = use_rm_only
    variant['_rm_extend_obs'] = extend_obs

    # Run MAPLE with RM
    # Note: The actual RM integration happens in the environment wrapper
    # which should be added in robosuite_launcher or via post-processing
    return maple_experiment(variant)


def main():
    parser = argparse.ArgumentParser(
        description='Run MAPLE with LTL Reward Machine'
    )

    # Task configuration
    available_tasks = get_available_tasks()
    parser.add_argument('--task', type=str, default='stack',
                        choices=available_tasks,
                        help=f'Task (available: {", ".join(available_tasks)})')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 500)')

    # RM configuration
    parser.add_argument('--no-rm', action='store_true',
                        help='Disable RM reward (baseline)')
    parser.add_argument('--rm-weight', type=float, default=1.0,
                        help='RM reward scale factor')
    parser.add_argument('--rm-only', action='store_true',
                        help='Use only RM reward (replace env reward)')
    parser.add_argument('--rm-extend-obs', action='store_true',
                        help='Extend observation with RM state one-hot')

    # RM reward shaping (based on IJCAI 2019 "LTL and Beyond")
    parser.add_argument('--rm-mode', type=str, default='hybrid',
                        choices=['sparse', 'distance', 'progression', 'hybrid'],
                        help='RM reward mode: sparse (terminal only), distance (-dist), '
                             'progression (transition rewards), hybrid (progression + potential)')
    parser.add_argument('--rm-potential', action='store_true', default=True,
                        help='Use potential-based shaping in hybrid mode (default: True)')
    parser.add_argument('--no-rm-potential', dest='rm_potential', action='store_false',
                        help='Disable potential-based shaping')
    parser.add_argument('--rm-potential-scale', type=float, default=0.1,
                        help='Scale for potential-based shaping reward (default: 0.1)')
    parser.add_argument('--rm-terminal-reward', type=float, default=1.0,
                        help='Bonus reward for reaching terminal state (default: 1.0)')

    # Parallelization
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel exploration environments')
    parser.add_argument('--dummy-vec-env', action='store_true',
                        help='Use DummyVecEnv (sequential)')

    # GPU configuration
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')

    # Experiment configuration
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--label', type=str, default=None,
                        help='Experiment label')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with minimal steps')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--min-steps', type=int, default=None,
                        help='Min steps before training')
    parser.add_argument('--replay-buffer-size', type=int, default=None,
                        help='Replay buffer size')
    parser.add_argument('--snapshot-gap', type=int, default=25,
                        help='Epochs between snapshots')

    args = parser.parse_args()

    # Build variant
    variant = get_base_variant()
    variant = process_variant(variant, args)

    # Experiment naming
    if args.no_rm:
        exp_prefix = args.label or "baseline"
    else:
        exp_prefix = args.label or f"rm{args.rm_weight}"

    # Print configuration
    print("=" * 60)
    print("MAPLE with LTL Reward Machine")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"RM enabled: {not args.no_rm}")
    if not args.no_rm:
        rm = create_rm(args.task, reward_mode=args.rm_mode,
                       use_potential_shaping=args.rm_potential,
                       potential_scale=args.rm_potential_scale,
                       terminal_reward=args.rm_terminal_reward)
        print(f"  RM name: {rm.name}")
        print(f"  States: {rm.n_states}")
        print(f"  Reward scale: {args.rm_weight}")
        print(f"  Use RM only: {args.rm_only}")
        print(f"  Extend obs: {args.rm_extend_obs}")
        print(f"  Reward mode: {args.rm_mode}")
        print(f"  Potential shaping: {args.rm_potential}")
        print(f"  Potential scale: {args.rm_potential_scale}")
        print(f"  Terminal reward: {args.rm_terminal_reward}")
        print(f"\n  Distance to terminal from u0: {rm.distance_to_terminal('u0')}")
        print(f"  Potential at u0: {rm.potential('u0'):.2f}")
    print(f"\nParallel envs: {args.num_envs}")
    print(f"Min steps: {variant['algorithm_kwargs']['min_num_steps_before_training']}")
    print(f"Debug mode: {args.debug}")
    print(f"GPU: {'disabled' if args.no_gpu else args.gpu}")
    print("=" * 60)
    print()

    # Run experiment
    run_experiment(
        rm_experiment,
        exp_folder=f"rm-maple-{args.task}",
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
