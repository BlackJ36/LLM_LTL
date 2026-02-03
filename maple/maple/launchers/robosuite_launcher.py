import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config

import maple.torch.pytorch_util as ptu
from maple.data_management.env_replay_buffer import EnvReplayBuffer
from maple.samplers.data_collector import MdpPathCollector, VectorizedMdpPathCollector
from maple.envs.vectorized_env import SubprocVecEnv, DummyVecEnv
from maple.torch.sac.policies import (
    TanhGaussianPolicy,
    PAMDPPolicy,
    MakeDeterministic
)
from maple.torch.sac.sac import SACTrainer
from maple.torch.sac.sac_hybrid import SACHybridTrainer
from maple.torch.networks import ConcatMlp
from maple.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import numpy as np
import torch

from maple.core.logging import logger

# Import RM wrapper for LTL reward machine integration
try:
    from llm_ltl.envs.rm_wrapper import RMEnvWrapper
    RM_WRAPPER_AVAILABLE = True
except ImportError:
    RM_WRAPPER_AVAILABLE = False

def experiment(variant):
    # GPU diagnostics
    print("\n" + "=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)
    print(f"ptu._use_gpu: {ptu._use_gpu}")
    print(f"ptu.device: {ptu.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")
    # Test tensor placement
    test_tensor = ptu.zeros(10)
    print(f"Test tensor device: {test_tensor.device}")

    # cuDNN benchmark mode for faster convolutions
    if variant.get('cudnn_benchmark', True) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"cuDNN benchmark: enabled")
    print("=" * 60 + "\n")

    # Initialize TensorBoard logging if enabled
    if variant.get('use_tensorboard', False):
        log_dir = logger.get_snapshot_dir()
        if log_dir is not None:
            logger.set_tensorboard(log_dir)
            logger.log("TensorBoard logging enabled. Log dir: {}".format(log_dir))

    def make_env(mode):
        assert mode in ['expl', 'eval']
        # Don't limit threads in main process - only limit in subprocess workers
        # torch.set_num_threads(1)  # Removed: this was limiting training performance

        env_variant = variant['env_variant']

        controller_config = load_controller_config(default_controller=env_variant['controller_type'])
        controller_config_update = env_variant.get('controller_config_update', {})
        controller_config.update(controller_config_update)

        robot_type = env_variant.get('robot_type', 'Panda')

        obs_keys = env_variant['robot_keys'] + env_variant['obj_keys']

        env = suite.make(
            env_name=env_variant['env_type'],
            robots=robot_type,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            controller_configs=controller_config,

            **env_variant['env_kwargs']
        )

        env = GymWrapper(env, keys=obs_keys)

        return env

    expl_env = make_env(mode='expl')
    eval_env = make_env(mode='eval')

    # RM (Reward Machine) configuration
    rm_instance = variant.get('_rm_instance')
    rm_events_fn = variant.get('_rm_events_fn')
    rm_reward_scale = variant.get('_rm_reward_scale', 1.0)
    rm_use_only = variant.get('_rm_use_only', False)
    rm_extend_obs = variant.get('_rm_extend_obs', False)
    rm_enabled = rm_instance is not None and rm_events_fn is not None and RM_WRAPPER_AVAILABLE

    # Wrap environments with RM wrapper if RM is enabled
    if rm_enabled:
        # Create fresh RM instances for each environment (to maintain separate states)
        from copy import deepcopy

        expl_rm = deepcopy(rm_instance)
        eval_rm = deepcopy(rm_instance)

        expl_env = RMEnvWrapper(
            expl_env, expl_rm, rm_events_fn,
            rm_reward_scale=rm_reward_scale,
            use_rm_reward_only=rm_use_only,
            extend_obs_with_rm=rm_extend_obs,
            log_rm_info=True
        )
        eval_env = RMEnvWrapper(
            eval_env, eval_rm, rm_events_fn,
            rm_reward_scale=rm_reward_scale,
            use_rm_reward_only=rm_use_only,
            extend_obs_with_rm=rm_extend_obs,
            log_rm_info=True
        )
        logger.log(f"RM Wrapper enabled: scale={rm_reward_scale}, rm_only={rm_use_only}")

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    action_dim_s = getattr(expl_env, "action_skill_dim", 0)
    action_dim_p = action_dim - action_dim_s
    if action_dim_s == 0:
        trainer_class = SACTrainer
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )

        target_entropy_config = variant['ll_sac_variant'].get('target_entropy_config', {})

        if variant['ll_sac_variant'].get('high_init_ent'):
            target_entropy_config.update(dict(
                init_epochs=200,
                init=0,
            ))

        variant['trainer_kwargs']['target_entropy_config'] = target_entropy_config

    else:
        trainer_class = SACHybridTrainer
        policy_kwargs = {}
        policy_class = PAMDPPolicy

        pamdp_variant = variant.get('pamdp_variant', {})

        for k in [
            'one_hot_s',
        ]:
            policy_kwargs[k] = pamdp_variant[k]

        for k in [
            'target_entropy_s',
            'target_entropy_p',
        ]:
            variant['trainer_kwargs'][k] = pamdp_variant.get(k, None)

        target_entropy_config = pamdp_variant.get('target_entropy_config', {})

        if pamdp_variant.get('high_init_ent'):
            assert pamdp_variant['one_hot_s']
            target_entropy_config['init_epochs'] = 200
            target_entropy_config['init_s'] = 0.97 * np.log(action_dim_s)
            if not pamdp_variant.get('disable_high_init_ent_p', False):
                target_entropy_config['init_p'] = 0

        if pamdp_variant.get('one_hot_factor'):
            target_entropy_config['one_hot_factor'] = pamdp_variant['one_hot_factor']

        variant['trainer_kwargs']['target_entropy_config'] = target_entropy_config

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim_s=action_dim_s,
            action_dim_p=action_dim_p,
            hidden_sizes=[M, M],
            **policy_kwargs
        )

    # Apply torch.compile optimization if enabled (PyTorch 2.x)
    if variant.get('use_torch_compile', False):
        compile_mode = variant.get('torch_compile_mode', 'reduce-overhead')
        print(f"\n[torch.compile] Applying with mode='{compile_mode}'...")
        qf1 = torch.compile(qf1, mode=compile_mode)
        qf2 = torch.compile(qf2, mode=compile_mode)
        target_qf1 = torch.compile(target_qf1, mode=compile_mode)
        target_qf2 = torch.compile(target_qf2, mode=compile_mode)
        policy = torch.compile(policy, mode=compile_mode)
        print(f"[torch.compile] Policy type: {type(policy).__module__}.{type(policy).__name__}")
        print(f"[torch.compile] QF1 type: {type(qf1).__module__}.{type(qf1).__name__}")
        print("[torch.compile] All networks compiled successfully\n")

    eval_policy = MakeDeterministic(policy)

    rollout_fn_kwargs = variant.get('rollout_fn_kwargs', {})

    # Check if vectorized training is enabled
    num_expl_envs = variant.get('num_expl_envs', 1)
    num_eval_envs = variant.get('num_eval_envs', 1)
    use_dummy_vec_env = variant.get('use_dummy_vec_env', False)

    # Preserve single eval_env for video recording
    eval_env_for_video = eval_env

    # Get RM config for subprocess creation (avoids pickling RM objects)
    rm_task_name = variant.get('rm_variant', {}).get('task_name', None)
    rm_config = {
        'enabled': rm_enabled,
        'task_name': rm_task_name,
        'reward_scale': rm_reward_scale,
        'use_only': rm_use_only,
        'extend_obs': rm_extend_obs,
    }

    # Evaluation environment parallelization
    if num_eval_envs > 1:
        def make_eval_env_fn():
            env = make_env(mode='eval')
            # Wrap with RM if enabled - create fresh RM in subprocess
            if rm_config['enabled'] and rm_config['task_name']:
                from llm_ltl.reward_machines import create_rm, get_events_fn
                rm = create_rm(rm_config['task_name'])
                events_fn = get_events_fn(rm_config['task_name'])
                env = RMEnvWrapper(
                    env, rm, events_fn,
                    rm_reward_scale=rm_config['reward_scale'],
                    use_rm_reward_only=rm_config['use_only'],
                    extend_obs_with_rm=rm_config['extend_obs'],
                    log_rm_info=True
                )
            return env

        eval_env_fns = [make_eval_env_fn for _ in range(num_eval_envs)]

        if use_dummy_vec_env:
            vec_eval_env = DummyVecEnv(eval_env_fns)
        else:
            vec_eval_env = SubprocVecEnv(eval_env_fns)

        eval_path_collector = VectorizedMdpPathCollector(
            vec_eval_env,
            eval_policy,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )
        # Store single env reference for video recording
        eval_path_collector._single_env = eval_env_for_video
    else:
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy,
            save_env_in_snapshot=False,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )

    # Exploration can use vectorized environments
    if num_expl_envs > 1:
        # Create vectorized exploration environment
        def make_expl_env_fn():
            env = make_env(mode='expl')
            # Wrap with RM if enabled - create fresh RM in subprocess
            if rm_config['enabled'] and rm_config['task_name']:
                from llm_ltl.reward_machines import create_rm, get_events_fn
                rm = create_rm(rm_config['task_name'])
                events_fn = get_events_fn(rm_config['task_name'])
                env = RMEnvWrapper(
                    env, rm, events_fn,
                    rm_reward_scale=rm_config['reward_scale'],
                    use_rm_reward_only=rm_config['use_only'],
                    extend_obs_with_rm=rm_config['extend_obs'],
                    log_rm_info=True
                )
            return env

        env_fns = [make_expl_env_fn for _ in range(num_expl_envs)]

        if use_dummy_vec_env:
            vec_expl_env = DummyVecEnv(env_fns)
        else:
            vec_expl_env = SubprocVecEnv(env_fns)

        expl_path_collector = VectorizedMdpPathCollector(
            vec_expl_env,
            policy,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )
        # Use vectorized env for replay buffer space info
        expl_env_for_buffer = expl_env
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
            save_env_in_snapshot=False,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )
        expl_env_for_buffer = expl_env
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    if 'ckpt_epoch' in variant:
        variant['algorithm_kwargs']['num_epochs'] = variant['ckpt_epoch']
        variant['algorithm_kwargs']['eval_epoch_freq'] = 1

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    if variant.get("save_video", True):
        variant['dump_video_kwargs']['imsize'] = eval_env.camera_heights[0]
        variant['dump_video_kwargs']['rollout_fn_kwargs'] = rollout_fn_kwargs
        video_save_func = get_video_save_func(variant)
        algorithm.post_epoch_funcs.append(video_save_func)

    if 'ckpt_path' in variant:
        ckpt_update_func = get_ckpt_update_func(variant)
        algorithm.pre_epoch_funcs.insert(0, ckpt_update_func)

    algorithm.to(ptu.device)
    algorithm.train(start_epoch=variant.get('ckpt_epoch', 0))

    # Print training summary table
    if variant.get('print_summary', True):
        print_training_summary()

def print_training_summary():
    """Print a formatted summary table of training progress."""
    import os
    import csv
    from maple.core.logging import logger

    log_dir = logger.get_snapshot_dir()
    if log_dir is None:
        return

    progress_file = os.path.join(log_dir, 'progress.csv')
    if not os.path.exists(progress_file):
        return

    try:
        with open(progress_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        # Define columns to display
        columns = [
            ('Epoch', 'Epoch'),
            ('Expl Returns', 'expl/Average Returns'),
            ('Eval Returns', 'eval/Average Returns'),
            ('Success', 'expl/env_infos/final/success Mean'),
            ('Epoch Time', 'time/epoch (s)'),
            ('Buffer Size', 'replay_buffer/size'),
        ]

        # Print header
        print("\n" + "=" * 90)
        print("Training Summary")
        print("=" * 90)

        # Build header row
        header = ""
        for name, _ in columns:
            header += f"{name:>15}"
        print(header)
        print("-" * 90)

        # Print data rows
        for row in rows:
            line = ""
            for name, key in columns:
                val = row.get(key, 'N/A')
                if val != 'N/A':
                    try:
                        num = float(val)
                        if name in ['Epoch', 'Buffer Size']:
                            line += f"{int(num):>15}"
                        elif name == 'Epoch Time':
                            line += f"{num:>14.1f}s"
                        elif name == 'Success':
                            line += f"{num*100:>14.1f}%"
                        else:
                            line += f"{num:>15.2f}"
                    except:
                        line += f"{val:>15}"
                else:
                    line += f"{'N/A':>15}"
            print(line)

        print("=" * 90)
        print(f"Log directory: {log_dir}")
        print("=" * 90 + "\n")

    except Exception as e:
        print(f"Could not print training summary: {e}")

def get_ckpt_update_func(variant):
    import os.path as osp
    import torch
    from maple.launchers.conf import LOCAL_LOG_DIR

    def ckpt_update_func(algo, epoch):
        if epoch == variant.get('ckpt_epoch', None) or epoch % algo._eval_epoch_freq == 0:
            filename = osp.join(LOCAL_LOG_DIR, variant['ckpt_path'], 'itr_%d.pkl' % epoch)
            try:
                print("Loading ckpt from", filename)
                if ptu.gpu_enabled():
                    data = torch.load(filename, map_location='cuda:0', weights_only=False)
                else:
                    data = torch.load(filename, map_location='cpu', weights_only=False)
                print("checkpoint loaded.")
            except FileNotFoundError:
                print('Could not locate checkpoint. Aborting.')
                exit()

            eval_policy = data['evaluation/policy']
            eval_policy.to(ptu.device)
            algo.eval_data_collector._policy = eval_policy

            expl_policy = data['exploration/policy']
            expl_policy.to(ptu.device)
            algo.expl_data_collector._policy = expl_policy

    return ckpt_update_func

def get_video_save_func(variant):
    from maple.samplers.rollout_functions import rollout
    from maple.launchers.visualization import dump_video

    save_period = variant.get('save_video_period', 50)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    dump_video_kwargs['horizon'] = variant['algorithm_kwargs']['max_path_length']

    def video_save_func(algo, epoch):
        if epoch % save_period == 0 or epoch == algo.num_epochs:
            # Skip exploration video for vectorized collectors (they don't have _env)
            if variant.get('vis_expl', True) and hasattr(algo.expl_data_collector, '_env'):
                dump_video(
                    algo.expl_data_collector._env,
                    algo.expl_data_collector._policy,
                    rollout,
                    mode='expl',
                    epoch=epoch,
                    **dump_video_kwargs
                )

            # Get eval environment (support both parallel and non-parallel cases)
            if hasattr(algo.eval_data_collector, '_single_env'):
                # Parallel evaluation: use saved single env for video recording
                eval_env_for_dump = algo.eval_data_collector._single_env
            else:
                # Non-parallel evaluation: use _env directly
                eval_env_for_dump = algo.eval_data_collector._env

            dump_video(
                eval_env_for_dump,
                algo.eval_data_collector._policy,
                rollout,
                mode='eval',
                epoch=epoch,
                **dump_video_kwargs
            )
    return video_save_func