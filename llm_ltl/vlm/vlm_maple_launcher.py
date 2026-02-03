"""
MAPLE launcher with VLM reward integration.

This launcher extends the standard MAPLE robosuite launcher
to add VLM-based reward signals for guiding action primitive selection.
"""

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config

import maple.torch.pytorch_util as ptu
from maple.data_management.env_replay_buffer import EnvReplayBuffer
from maple.data_management.vlm_prioritized_replay_buffer import VLMPrioritizedEnvReplayBuffer
from maple.samplers.data_collector import MdpPathCollector, VectorizedMdpPathCollector
from maple.envs.vectorized_env import SubprocVecEnv, DummyVecEnv
from maple.torch.sac.policies import (
    TanhGaussianPolicy,
    PAMDPPolicy,
    MakeDeterministic
)
from maple.torch.sac.sac import SACTrainer
from maple.torch.sac.sac_hybrid import SACHybridTrainer
from maple.torch.sac.sac_hybrid_vlm import SACHybridVLMTrainer
from maple.torch.networks import ConcatMlp
from maple.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from maple.torch.vlm_batch_rl_algorithm import VLMBatchRLAlgorithm
from maple.core.logging import logger

from llm_ltl.vlm import QwenVLClient, VLMRewardWrapper
from llm_ltl.vlm.vlm_reward_wrapper import VLMRewardWrapperAsync

import numpy as np
import torch


class VLMGymWrapper(GymWrapper):
    """
    Extended GymWrapper that preserves skill_controller access
    and adds VLM reward functionality.
    """

    def __init__(
        self,
        env,
        keys=None,
        vlm_config=None,
    ):
        super().__init__(env, keys=keys)
        self.vlm_config = vlm_config or {}
        self._vlm_wrapper = None

    def enable_vlm_reward(
        self,
        task_description: str,
        vlm_client=None,
        vlm_reward_scale: float = 0.1,
        eval_frequency: int = 1,
        use_async: bool = True,
        **kwargs
    ):
        """Enable VLM reward for this environment."""
        if vlm_client is None:
            vlm_client = QwenVLClient(**self.vlm_config.get('client_kwargs', {}))

        # Get skill controller from wrapped env
        skill_controller = getattr(self.env, 'skill_controller', None)

        wrapper_class = VLMRewardWrapperAsync if use_async else VLMRewardWrapper

        self._vlm_wrapper = wrapper_class(
            env=self,
            task_description=task_description,
            skill_controller=skill_controller,
            vlm_client=vlm_client,
            vlm_reward_scale=vlm_reward_scale,
            eval_frequency=eval_frequency,
            **kwargs
        )
        return self._vlm_wrapper


def experiment(variant):
    """
    Run MAPLE experiment with optional VLM reward.

    Additional variant keys for VLM:
        vlm_variant:
            enabled: bool - Whether to use VLM reward
            task_description: str - Natural language task description
            api_base: str - VLM API endpoint
            model_name: str - VLM model name
            vlm_reward_scale: float - Scale for VLM reward
            eval_frequency: int - VLM evaluation frequency
            use_async: bool - Use async VLM evaluation
            binary_weight: float - Weight for binary judgment
            progress_weight: float - Weight for progress evaluation
    """
    # Initialize TensorBoard logging if enabled
    if variant.get('use_tensorboard', False):
        log_dir = logger.get_snapshot_dir()
        if log_dir is not None:
            logger.set_tensorboard(log_dir)
            logger.log("TensorBoard logging enabled. Log dir: {}".format(log_dir))

    # VLM configuration
    vlm_variant = variant.get('vlm_variant', {})
    vlm_enabled = vlm_variant.get('enabled', False)

    if vlm_enabled:
        vlm_client = QwenVLClient(
            api_base=vlm_variant.get('api_base', 'http://172.19.1.40:8001'),
            model_name=vlm_variant.get('model_name', 'Qwen/Qwen3-VL-8B-Instruct'),
        )
        logger.log("VLM reward enabled with model: {}".format(vlm_variant.get('model_name')))
    else:
        vlm_client = None

    def make_env(mode):
        assert mode in ['expl', 'eval']
        torch.set_num_threads(1)

        env_variant = variant['env_variant']

        controller_config = load_controller_config(
            default_controller=env_variant['controller_type']
        )
        controller_config_update = env_variant.get('controller_config_update', {})
        controller_config.update(controller_config_update)

        robot_type = env_variant.get('robot_type', 'Panda')
        obs_keys = env_variant['robot_keys'] + env_variant['obj_keys']

        env = suite.make(
            env_name=env_variant['env_type'],
            robots=robot_type,
            has_renderer=False,
            has_offscreen_renderer=True,  # Required for VLM
            use_camera_obs=False,
            controller_configs=controller_config,
            **env_variant['env_kwargs']
        )

        env = GymWrapper(env, keys=obs_keys)

        # Wrap with VLM reward if enabled
        # For eval mode, only wrap if save_vlm_results is True (for VLM evaluation)
        should_wrap_vlm = vlm_enabled and (
            mode == 'expl' or vlm_variant.get('save_vlm_results', False)
        )
        if should_wrap_vlm:
            skill_controller = getattr(env.env, 'skill_controller', None)

            # Use sync mode for eval to ensure all VLM results are captured
            use_async = vlm_variant.get('use_async', True) and mode == 'expl'
            wrapper_class = VLMRewardWrapperAsync if use_async else VLMRewardWrapper

            env = wrapper_class(
                env=env,
                task_description=vlm_variant.get(
                    'task_description',
                    'Complete the manipulation task'
                ),
                skill_controller=skill_controller,
                vlm_client=vlm_client,
                vlm_reward_scale=vlm_variant.get('vlm_reward_scale', 0.5),
                eval_frequency=vlm_variant.get('eval_frequency', 1),
                # New simplified parameters (B+C+D plan)
                action_weight=vlm_variant.get('action_weight', 0.2),  # Lower: MAPLE handles action selection
                progress_weight=vlm_variant.get('progress_weight', 0.8),  # Higher: encourages effective trajectories
                camera_name=vlm_variant.get('camera_name', 'frontview'),
                camera_height=vlm_variant.get('camera_height', 256),
                camera_width=vlm_variant.get('camera_width', 256),
                warmup_steps=vlm_variant.get('warmup_steps', 0),
                image_history_size=vlm_variant.get('image_history_size', 4),
                # Plan B: Potential-based shaping
                use_potential_shaping=vlm_variant.get('use_potential_shaping', True),
                gamma=vlm_variant.get('gamma', 0.99),
                # Plan D: Curriculum learning
                curriculum_mode=vlm_variant.get('curriculum_mode', 'none'),
                curriculum_start=vlm_variant.get('curriculum_start', 1.0),
                curriculum_end=vlm_variant.get('curriculum_end', 0.1),
            )

        return env

    expl_env = make_env(mode='expl')
    eval_env = make_env(mode='eval')

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

        for k in ['one_hot_s']:
            policy_kwargs[k] = pamdp_variant[k]

        for k in ['target_entropy_s', 'target_entropy_p']:
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

    eval_policy = MakeDeterministic(policy)

    rollout_fn_kwargs = variant.get('rollout_fn_kwargs', {})

    # Check if vectorized training is enabled
    num_expl_envs = variant.get('num_expl_envs', 1)
    num_eval_envs = variant.get('num_eval_envs', 1)
    use_dummy_vec_env = variant.get('use_dummy_vec_env', False)

    # Evaluation path collector (usually single env)
    if num_eval_envs > 1:
        def make_eval_env_fn():
            return make_env(mode='eval')

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
    else:
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy,
            save_env_in_snapshot=False,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )

    # Exploration path collector (can be vectorized)
    if num_expl_envs > 1:
        def make_expl_env_fn():
            return make_env(mode='expl')

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
        logger.log(f"Using {num_expl_envs} parallel exploration environments")
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
            save_env_in_snapshot=False,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )

    # Check if VLM prioritized replay is enabled
    vlm_priority_config = vlm_variant.get('priority_sampling', {})
    use_vlm_priority = vlm_enabled and vlm_priority_config.get('enabled', False)

    if use_vlm_priority:
        priority_mode = vlm_priority_config.get('mode', 'hybrid')
        replay_buffer = VLMPrioritizedEnvReplayBuffer(
            max_replay_buffer_size=variant['replay_buffer_size'],
            env=expl_env,
            alpha=vlm_priority_config.get('alpha', 0.6),
            beta_start=vlm_priority_config.get('beta_start', 0.4),
            beta_end=vlm_priority_config.get('beta_end', 1.0),
            beta_anneal_steps=vlm_priority_config.get('beta_anneal_steps', 100000),
            # Simplified: action_score + progress (no confidence)
            action_score_weight=vlm_priority_config.get('action_score_weight', 0.2),
            progress_weight=vlm_priority_config.get('progress_weight', 0.8),
            priority_mode=priority_mode,
            vlm_weight=vlm_priority_config.get('vlm_weight', 0.5),
            td_weight=vlm_priority_config.get('td_weight', 0.5),
            # Stability parameters to prevent divergence
            max_is_weight=vlm_priority_config.get('max_is_weight', 10.0),
            max_td_error=vlm_priority_config.get('max_td_error', 10.0),
            # Buffer filtering (reduce ineffective trajectories)
            min_progress_threshold=vlm_priority_config.get('min_progress_threshold', 0.0),
        )
        logger.log("Using VLM-Prioritized Replay Buffer")
        logger.log(f"  Mode: {priority_mode}")
        if priority_mode == 'hybrid':
            logger.log(f"  Hybrid weights: VLM={vlm_priority_config.get('vlm_weight', 0.5)}, "
                       f"TD={vlm_priority_config.get('td_weight', 0.5)}")
        logger.log(f"  VLM weights: action_score={vlm_priority_config.get('action_score_weight', 0.2)}, "
                   f"progress={vlm_priority_config.get('progress_weight', 0.8)}")
        logger.log(f"  Stability: max_is_weight={vlm_priority_config.get('max_is_weight', 10.0)}, "
                   f"max_td_error={vlm_priority_config.get('max_td_error', 10.0)}")
    else:
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )

    # Use VLM-aware trainer if prioritized replay is enabled
    if use_vlm_priority and action_dim_s > 0:
        trainer = SACHybridVLMTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            use_importance_sampling=True,
            **variant['trainer_kwargs']
        )
        logger.log("Using SACHybridVLMTrainer with importance sampling")
    else:
        trainer = trainer_class(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )

    # Use VLM-aware algorithm if priority sampling is enabled
    if use_vlm_priority:
        # Early stopping configuration
        early_stop_config = vlm_variant.get('early_stop', {})
        algorithm = VLMBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            update_priorities=True,
            early_stop_enabled=early_stop_config.get('enabled', True),
            early_stop_success_threshold=early_stop_config.get('threshold', 0.95),
            early_stop_patience=early_stop_config.get('patience', 3),
            **variant['algorithm_kwargs']
        )
        logger.log("Using VLMBatchRLAlgorithm with priority updates")
        if early_stop_config.get('enabled', True):
            logger.log(f"  Early stopping: threshold={early_stop_config.get('threshold', 0.95)}, "
                      f"patience={early_stop_config.get('patience', 3)}")
    else:
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algorithm_kwargs']
        )

    algorithm.to(ptu.device)
    algorithm.train()


# Example configuration for running with VLM reward (B+C+D plan)
EXAMPLE_VLM_CONFIG = {
    'vlm_variant': {
        'enabled': True,
        'task_description': 'Pick up the cube and lift it above the table',
        'api_base': 'http://172.19.1.40:8001',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
        'vlm_reward_scale': 0.5,   # 0.3-1.0 recommended for strong VLM guidance
        'eval_frequency': 1,        # Every step
        'use_async': True,
        # Reward weights (Plan C: soft scoring)
        'action_weight': 0.2,       # Lower: MAPLE's dual-policy already learns action selection
        'progress_weight': 0.8,     # Higher: encourages sampling of effective trajectories
        'camera_name': 'frontview',
        'camera_height': 256,
        'camera_width': 256,
        # Image history: VLM sees last N frames + current (total N+1 images)
        'image_history_size': 4,    # 4 historical + 1 current = 5 images
        # Plan B: Potential-based progress shaping
        'use_potential_shaping': True,  # r = γΦ(s') - Φ(s), allows negative progress reward
        'gamma': 0.99,                  # Discount factor for potential shaping
        # Plan D: Curriculum learning (VLM weight decay)
        'curriculum_mode': 'linear',    # 'none', 'linear', 'step', 'cosine'
        'curriculum_start': 1.0,        # Initial VLM weight
        'curriculum_end': 0.1,          # Final VLM weight

        # VLM + TD Hybrid Prioritized Experience Replay
        # Combines VLM evaluation (decision quality) with TD-error (learning signal)
        'priority_sampling': {
            'enabled': True,         # Enable hybrid prioritized replay
            'mode': 'hybrid',        # 'vlm_only', 'td_only', or 'hybrid'
            'alpha': 0.6,            # Priority exponent (0=uniform, 1=full priority)
            'beta_start': 0.4,       # IS correction start (0=no correction)
            'beta_end': 1.0,         # IS correction end (1=full correction)
            'beta_anneal_steps': 100000,
            # VLM score composition for priority (action_score + progress)
            'action_score_weight': 0.2,  # Weight for action score
            'progress_weight': 0.8,      # Weight for task progress
            # Hybrid mode weights (VLM vs TD-error):
            'vlm_weight': 0.5,          # VLM contribution to priority
            'td_weight': 0.5,           # TD-error contribution to priority
            # Buffer filtering (reduce ineffective trajectories)
            'min_progress_threshold': 0.0,  # Only store samples with progress >= threshold
        },
    }
}


if __name__ == '__main__':
    # Example usage
    print("VLM-MAPLE Launcher")
    print("=" * 50)
    print("\nExample VLM configuration:")
    import json
    print(json.dumps(EXAMPLE_VLM_CONFIG, indent=2))
    print("\nTo use, merge this config with your MAPLE variant config.")
