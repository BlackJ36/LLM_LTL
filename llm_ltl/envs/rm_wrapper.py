"""
Reward Machine environment wrapper.

Wraps any gym environment to add RM-based rewards.
"""

from typing import Any, Callable, Dict, Optional, Set, Tuple
from collections import OrderedDict
import numpy as np
import gym


class RMEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds Reward Machine rewards to an environment.

    The wrapper:
    1. Detects events from observations/info using the provided events function
    2. Updates the RM state based on detected events
    3. Computes RM reward for state transitions
    4. Optionally extends observation with RM state (one-hot)
    5. Logs RM diagnostics to info dict

    Example usage:
        from llm_ltl.reward_machines import create_rm, get_events_fn
        from llm_ltl.envs import RMEnvWrapper

        rm = create_rm('stack')
        get_events = get_events_fn('stack')
        wrapped_env = RMEnvWrapper(env, rm, get_events, rm_reward_scale=1.0)
    """

    def __init__(
        self,
        env: gym.Env,
        reward_machine,
        get_events_fn: Callable[[Dict, Dict], Set[str]],
        rm_reward_scale: float = 1.0,
        use_rm_reward_only: bool = False,
        extend_obs_with_rm: bool = False,
        log_rm_info: bool = True,
    ):
        """
        Initialize the RM wrapper.

        Args:
            env: The environment to wrap
            reward_machine: RewardMachine instance
            get_events_fn: Function that takes (obs, info) and returns Set[str]
            rm_reward_scale: Scale factor for RM rewards
            use_rm_reward_only: If True, replace env reward with RM reward only
            extend_obs_with_rm: If True, append RM state one-hot to observation
            log_rm_info: If True, add RM info to step info dict
        """
        super().__init__(env)

        self.rm = reward_machine
        self.get_events = get_events_fn
        self.rm_reward_scale = rm_reward_scale
        self.use_rm_reward_only = use_rm_reward_only
        self.extend_obs_with_rm = extend_obs_with_rm
        self.log_rm_info = log_rm_info

        # Statistics
        self._episode_rm_reward = 0.0
        self._episode_transitions = 0
        self._total_episodes = 0

        # Modify observation space if extending with RM state
        if extend_obs_with_rm:
            self._setup_extended_obs_space()

    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        # This is called when attribute is not found in the wrapper
        return getattr(self.env, name)

    def _setup_extended_obs_space(self):
        """Set up extended observation space with RM state."""
        original_space = self.env.observation_space

        if isinstance(original_space, gym.spaces.Box):
            # Extend Box space
            low = np.concatenate([
                original_space.low.flatten(),
                np.zeros(self.rm.n_states)
            ])
            high = np.concatenate([
                original_space.high.flatten(),
                np.ones(self.rm.n_states)
            ])
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float32
            )
        elif isinstance(original_space, gym.spaces.Dict):
            # Add new key to Dict space
            spaces = dict(original_space.spaces)
            spaces['rm_state'] = gym.spaces.Box(
                low=0, high=1, shape=(self.rm.n_states,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            # Keep original for other space types
            pass

    def reset(self, **kwargs):
        """Reset environment and RM."""
        obs = self.env.reset(**kwargs)

        # Reset RM
        self.rm.reset()

        # Reset statistics
        self._episode_rm_reward = 0.0
        self._episode_transitions = 0

        # Extend observation if needed
        if self.extend_obs_with_rm:
            obs = self._extend_obs(obs)

        return obs

    def step(self, action):
        """Step environment and update RM."""
        obs, env_reward, done, info = self.env.step(action)

        # Detect events
        events = self.get_events(obs, info)

        # Update RM
        old_state = self.rm.current_state
        new_state, rm_reward, rm_terminal = self.rm.step(events)

        # Track transitions
        if old_state != new_state:
            self._episode_transitions += 1

        # Scale RM reward
        scaled_rm_reward = rm_reward * self.rm_reward_scale
        self._episode_rm_reward += scaled_rm_reward

        # Compute total reward
        if self.use_rm_reward_only:
            total_reward = scaled_rm_reward
        else:
            total_reward = env_reward + scaled_rm_reward

        # Extend observation if needed
        if self.extend_obs_with_rm:
            obs = self._extend_obs(obs)

        # Add RM info
        if self.log_rm_info:
            info['rm_state'] = new_state
            info['rm_state_idx'] = self.rm.get_state_index()
            info['rm_reward'] = rm_reward
            info['rm_reward_scaled'] = scaled_rm_reward
            info['rm_events'] = events
            info['rm_terminal'] = rm_terminal
            info['rm_progress'] = self.rm.get_progress()
            info['rm_episode_reward'] = self._episode_rm_reward
            info['rm_episode_transitions'] = self._episode_transitions

        # Optionally terminate on RM terminal
        # (usually we don't, as env has its own termination)
        # if rm_terminal:
        #     done = True

        if done:
            self._total_episodes += 1

        return obs, total_reward, done, info

    def _extend_obs(self, obs):
        """Extend observation with RM state one-hot."""
        rm_one_hot = self.rm.get_state_one_hot()

        if isinstance(obs, np.ndarray):
            return np.concatenate([obs.flatten(), rm_one_hot])
        elif isinstance(obs, dict):
            obs = dict(obs)
            obs['rm_state'] = rm_one_hot
            return obs
        else:
            return obs

    def get_rm_diagnostics(self) -> OrderedDict:
        """Get RM diagnostics for logging."""
        diag = self.rm.get_diagnostics()
        diag['rm_wrapper/total_episodes'] = self._total_episodes
        diag['rm_wrapper/episode_rm_reward'] = self._episode_rm_reward
        diag['rm_wrapper/episode_transitions'] = self._episode_transitions
        return diag


class VectorRMEnvWrapper:
    """
    Wrapper for vectorized environments with per-env RMs.

    Each environment in the vector has its own RM instance.
    """

    def __init__(
        self,
        vec_env,
        reward_machine_class,
        get_events_fn: Callable[[Dict, Dict], Set[str]],
        rm_reward_scale: float = 1.0,
        use_rm_reward_only: bool = False,
        log_rm_info: bool = True,
    ):
        """
        Initialize vectorized RM wrapper.

        Args:
            vec_env: Vectorized environment
            reward_machine_class: Callable that creates a new RM instance
            get_events_fn: Event detection function
            rm_reward_scale: Scale factor for RM rewards
            use_rm_reward_only: If True, use only RM reward
            log_rm_info: If True, log RM info
        """
        self.vec_env = vec_env
        self.n_envs = getattr(vec_env, 'num_envs', 1)
        self.get_events = get_events_fn
        self.rm_reward_scale = rm_reward_scale
        self.use_rm_reward_only = use_rm_reward_only
        self.log_rm_info = log_rm_info

        # Create RM for each environment
        self.rms = [reward_machine_class() for _ in range(self.n_envs)]

        # Forward common attributes
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space

    def reset(self, **kwargs):
        """Reset all environments and RMs."""
        obs = self.vec_env.reset(**kwargs)
        for rm in self.rms:
            rm.reset()
        return obs

    def step(self, actions):
        """Step all environments and update RMs."""
        obs, rewards, dones, infos = self.vec_env.step(actions)

        rm_rewards = np.zeros(self.n_envs)

        for i in range(self.n_envs):
            # Get info for this env
            info_i = infos[i] if isinstance(infos, list) else infos

            # Detect events (need to extract per-env obs)
            if isinstance(obs, dict):
                obs_i = {k: v[i] for k, v in obs.items()}
            else:
                obs_i = obs[i]

            events = self.get_events(obs_i, info_i)

            # Update RM
            _, rm_reward, _ = self.rms[i].step(events)
            rm_rewards[i] = rm_reward * self.rm_reward_scale

            # Reset RM on done
            if dones[i]:
                self.rms[i].reset()

            # Add RM info
            if self.log_rm_info and isinstance(infos, list):
                infos[i]['rm_reward'] = rm_reward
                infos[i]['rm_state'] = self.rms[i].current_state

        # Compute total rewards
        if self.use_rm_reward_only:
            total_rewards = rm_rewards
        else:
            total_rewards = rewards + rm_rewards

        return obs, total_rewards, dones, infos

    def __getattr__(self, name):
        """Forward attribute access to wrapped vec_env."""
        return getattr(self.vec_env, name)
