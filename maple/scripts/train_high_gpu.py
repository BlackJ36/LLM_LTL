"""
高 GPU 利用率训练脚本

策略:
1. 增大 batch_size - 让每次 GPU 计算更充分
2. 增加 num_trains_per_train_loop - 每次采样后多次训练
3. 增大网络层 - 更多 GPU 计算
4. 使用更大的 replay buffer - 更好的数据利用
"""
import argparse
from maple.launchers.launcher_util import run_experiment
from maple.launchers.robosuite_launcher import experiment
import maple.util.hyperparameter as hyp
import collections.abc

# 高 GPU 利用率配置
base_variant = dict(
    # 更大的网络 (256 -> 512)
    layer_size=512,

    # 更大的 replay buffer
    replay_buffer_size=int(2E6),

    rollout_fn_kwargs=dict(
        terminals_all_false=True,
    ),

    algorithm_kwargs=dict(
        num_epochs=10000,
        # 每轮采样步数
        num_expl_steps_per_train_loop=1000,  # 减少采样，增加训练比例
        num_eval_steps_per_epoch=1000,

        # 关键: 每次采样后训练更多次 (原来1000, 现在4000)
        num_trains_per_train_loop=4000,

        min_num_steps_before_training=10000,
        max_path_length=150,

        # 更大的 batch size (1024 -> 4096)
        batch_size=4096,

        eval_epoch_freq=10,
    ),

    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=1e-3,
        target_update_period=1,
        policy_lr=3e-5,
        qf_lr=3e-5,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),

    ll_sac_variant=dict(
        high_init_ent=True,
    ),

    pamdp_variant=dict(
        one_hot_s=True,
        high_init_ent=True,
        one_hot_factor=0.50,
    ),

    env_variant=dict(
        robot_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
        obj_keys=['object-state'],
        controller_type='OSC_POSITION_YAW',
        controller_config_update=dict(
            position_limits=[
                [-0.30, -0.30, 0.75],
                [0.15, 0.30, 1.15]
            ],
        ),
        env_kwargs=dict(
            ignore_done=True,
            reward_shaping=True,
            hard_reset=False,
            control_freq=10,
            camera_heights=512,
            camera_widths=512,
            table_offset=[-0.075, 0, 0.8],
            reward_scale=5.0,

            skill_config=dict(
                skills=['atomic', 'open', 'reach', 'grasp', 'push'],
                aff_penalty_fac=15.0,

                base_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.95]
                    ],
                    lift_height=0.95,
                    binary_gripper=True,
                    aff_threshold=0.06,
                    aff_type='dense',
                    aff_tanh_scaling=10.0,
                ),
                atomic_config=dict(
                    use_ori_params=True,
                ),
                reach_config=dict(
                    use_gripper_params=False,
                    local_xyz_scale=[0.0, 0.0, 0.06],
                    use_ori_params=False,
                    max_ac_calls=15,
                ),
                grasp_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    aff_threshold=0.03,
                    local_xyz_scale=[0.0, 0.0, 0.0],
                    use_ori_params=True,
                    max_ac_calls=20,
                    num_reach_steps=2,
                    num_grasp_steps=3,
                ),
                push_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    delta_xyz_scale=[0.25, 0.25, 0.05],
                    max_ac_calls=20,
                    use_ori_params=True,
                    aff_threshold=[0.12, 0.12, 0.04],
                ),
            ),
        ),
    ),
    save_video=False,  # 关闭视频以提高速度
    save_video_period=100,
    dump_video_kwargs=dict(
        rows=1,
        columns=6,
        pad_length=5,
        pad_color=0,
    ),
)

env_params = dict(
    lift={
        'env_variant.env_type': ['Lift'],
    },
    door={
        'env_variant.env_type': ['Door'],
        'env_variant.controller_type': ['OSC_POSITION'],
    },
    pnp={
        'env_variant.env_type': ['PickPlaceCan'],
    },
    stack={
        'env_variant.env_type': ['Stack'],
    },
)


def process_variant(variant, args):
    if args.debug:
        variant['algorithm_kwargs']['num_epochs'] = 5
        variant['algorithm_kwargs']['batch_size'] = 2048
        variant['algorithm_kwargs']['max_path_length'] = 50
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 50
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 200
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 200
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 500
        variant['replay_buffer_size'] = int(1E4)

    # 自定义参数覆盖
    if args.batch_size:
        variant['algorithm_kwargs']['batch_size'] = args.batch_size
    if args.num_trains:
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains
    if args.layer_size:
        variant['layer_size'] = args.layer_size

    variant['exp_label'] = args.label
    return variant


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='lift')
    parser.add_argument('--label', type=str, default='high_gpu')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--snapshot_gap', type=int, default=25)

    # GPU 利用率参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (默认4096)')
    parser.add_argument('--num_trains', type=int, default=None,
                       help='每轮训练次数 (默认4000)')
    parser.add_argument('--layer_size', type=int, default=None,
                       help='网络层大小 (默认512)')

    args = parser.parse_args()

    print("=" * 60)
    print("高 GPU 利用率训练配置")
    print("=" * 60)
    print(f"  Batch Size: {args.batch_size or 4096}")
    print(f"  训练次数/轮: {args.num_trains or 4000}")
    print(f"  网络层大小: {args.layer_size or 512}")
    print("=" * 60)

    search_space = env_params.get(args.env, {'env_variant.env_type': [args.env.capitalize()]})

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=base_variant,
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = process_variant(variant, args)

        run_experiment(
            experiment,
            exp_folder=args.env,
            exp_prefix=args.label,
            variant=variant,
            snapshot_mode='gap_and_last',
            snapshot_gap=args.snapshot_gap,
            exp_id=exp_id,
            use_gpu=(not args.no_gpu),
            gpu_id=args.gpu_id,
            mode='local',
            num_exps_per_instance=1,
        )
