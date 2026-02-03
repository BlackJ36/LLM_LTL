"""
Vectorized path collector for parallel environment sampling.
"""
from collections import deque, OrderedDict

import numpy as np

from maple.core.eval_util import create_stats_ordered_dict
from maple.samplers.data_collector.base import PathCollector
from maple.samplers.rollout_functions import vectorized_rollout


class VectorizedMdpPathCollector(PathCollector):
    """Path collector that uses vectorized environments for parallel sampling.

    This collector manages multiple environments running in parallel,
    collecting paths more efficiently by batching action inference on GPU.

    Attributes:
        _vec_env: Vectorized environment (SubprocVecEnv or DummyVecEnv)
        _policy: Policy for action selection
        _num_envs: Number of parallel environments
    """

    def __init__(
            self,
            vec_env,
            policy,
            max_num_epoch_paths_saved=None,
            rollout_fn_kwargs=None,
            save_env_in_snapshot=False,
    ):
        """Initialize vectorized path collector.

        Args:
            vec_env: Vectorized environment instance
            policy: Policy for action selection
            max_num_epoch_paths_saved: Maximum paths to keep in memory per epoch
            rollout_fn_kwargs: Additional kwargs for vectorized_rollout
            save_env_in_snapshot: Whether to save env in snapshot (disabled for vec envs)
        """
        self._vec_env = vec_env
        self._policy = policy
        self._num_envs = vec_env.num_envs

        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

        if rollout_fn_kwargs is None:
            rollout_fn_kwargs = {}
        self._rollout_fn_kwargs = rollout_fn_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._num_actions_total = 0

        # Vectorized envs are not serializable, so never save in snapshot
        self._save_env_in_snapshot = False

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths=True,
    ):
        """Collect paths using vectorized rollout.

        Uses parallel environments to collect paths until num_steps
        is reached. Each call to vectorized_rollout produces num_envs paths.

        Args:
            max_path_length: Maximum steps per episode
            num_steps: Minimum total steps to collect
            discard_incomplete_paths: If True, may discard partial paths

        Returns:
            List of collected path dicts
        """
        paths = []
        num_steps_collected = 0
        num_actions_collected = 0

        while num_steps_collected < num_steps:
            # Calculate remaining steps needed
            remaining_steps = num_steps - num_steps_collected

            # Adjust max_path_length if we're near the target
            max_path_length_this_loop = min(max_path_length, remaining_steps)

            # Skip if remaining steps too small for meaningful collection
            if discard_incomplete_paths and max_path_length_this_loop < max_path_length // 2:
                break

            # Collect paths from all parallel environments
            new_paths = vectorized_rollout(
                self._vec_env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                **self._rollout_fn_kwargs
            )

            # Accumulate statistics
            for path in new_paths:
                num_steps_collected += path['path_length']
                num_actions_collected += path['path_length_actions']

            paths.extend(new_paths)

        # Update totals
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._num_actions_total += num_actions_collected
        self._epoch_paths.extend(paths)

        return paths

    def get_epoch_paths(self):
        """Return paths collected in current epoch."""
        return self._epoch_paths

    def end_epoch(self, epoch):
        """Clear epoch paths for next epoch."""
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        """Return diagnostics dict for logging."""
        path_lens = [path['path_length'] for path in self._epoch_paths]

        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
            ('num actions total', self._num_actions_total),
            ('num parallel envs', self._num_envs),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        """Return snapshot dict for checkpointing."""
        return dict(
            policy=self._policy,
        )

    @property
    def num_envs(self):
        """Number of parallel environments."""
        return self._num_envs
