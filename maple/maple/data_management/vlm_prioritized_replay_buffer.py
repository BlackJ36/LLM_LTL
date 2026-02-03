"""
VLM-Prioritized Replay Buffer.

Uses VLM evaluation scores as priority weights for sampling.

VLM provides two signals:
1. action_score: How appropriate was the primitive selection (0-1)
2. progress: Task completion progress (0-1)

Priority encourages sampling of:
- Effective trajectories (high progress)
- Reasonable action choices (high action_score)
"""

import numpy as np
import warnings
from collections import OrderedDict
from gym.spaces import Discrete

from maple.data_management.simple_replay_buffer import SimpleReplayBuffer
from maple.envs.env_utils import get_dim


class SumTree:
    """
    Binary sum tree for O(log n) priority sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, priority, data_idx):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class VLMPrioritizedReplayBuffer(SimpleReplayBuffer):
    """
    Replay buffer with hybrid VLM + TD-error prioritized sampling.

    Combines two complementary signals:
    1. VLM evaluation: How GOOD was this action choice? (decision quality)
    2. TD-error: How SURPRISING was this outcome? (learning signal)

    Priority modes:
    - "vlm_only": priority = vlm_score^α
    - "td_only": priority = |td_error|^α (standard PER)
    - "hybrid": priority = vlm_score^(α×vlm_weight) × |td_error|^(α×td_weight)

    The hybrid mode learns from samples that are both:
    - Good decisions (high VLM score) - worth imitating
    - Surprising outcomes (high TD-error) - need more learning
    """

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes=None,
        # Priority parameters
        alpha=0.6,                    # Priority exponent
        beta_start=0.4,               # IS weight start
        beta_end=1.0,                 # IS weight end
        beta_anneal_steps=100000,
        epsilon=0.01,                 # Minimum priority
        # VLM score weights (within VLM component)
        action_score_weight=0.2,      # Weight for action score (lower: MAPLE handles action selection)
        progress_weight=0.8,          # Weight for progress (higher: encourages effective trajectories)
        # VLM keys in env_info
        action_score_key='vlm_action_score',
        progress_key='vlm_progress',
        # Hybrid mode settings
        priority_mode='hybrid',       # 'vlm_only', 'td_only', 'hybrid'
        vlm_weight=0.5,               # VLM contribution in hybrid mode
        td_weight=0.5,                # TD contribution in hybrid mode
        # Stability parameters (prevent divergence)
        max_is_weight=10.0,           # Clip IS weights to prevent extreme updates
        max_td_error=10.0,            # Clip TD-errors to prevent priority explosion
        # Buffer filtering (reduce ineffective trajectories)
        min_progress_threshold=0.0,   # Only store samples with progress >= threshold
        **kwargs
    ):
        if env_info_sizes is None:
            env_info_sizes = {}

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=observation_dim,
            action_dim=action_dim,
            env_info_sizes=env_info_sizes,
            **kwargs
        )

        # Priority parameters
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon

        # VLM score weights (should sum to 1.0 for interpretability)
        self.action_score_weight = action_score_weight
        self.progress_weight = progress_weight

        # VLM keys
        self.action_score_key = action_score_key
        self.progress_key = progress_key

        # Hybrid mode settings
        self.priority_mode = priority_mode
        self.vlm_weight = vlm_weight
        self.td_weight = td_weight

        # Stability parameters
        self.max_is_weight = max_is_weight
        self.max_td_error = max_td_error

        # Buffer filtering
        self.min_progress_threshold = min_progress_threshold
        self._filtered_count = 0  # Track filtered samples

        # Priority tree
        self.tree = SumTree(max_replay_buffer_size)
        self.max_priority = 1.0

        # VLM scores storage
        self._vlm_action_score = np.zeros(max_replay_buffer_size)
        self._vlm_progress = np.zeros(max_replay_buffer_size)
        self._vlm_priority_score = np.zeros(max_replay_buffer_size)  # Combined score for priority

        # TD-error storage (for hybrid mode)
        self._td_errors = np.zeros(max_replay_buffer_size)
        self._has_td_error = np.zeros(max_replay_buffer_size, dtype=bool)

        # Combined priority storage
        self._priorities = np.zeros(max_replay_buffer_size)

        # Step counter for beta annealing
        self._sample_step = 0

    @property
    def beta(self):
        """Annealed importance sampling exponent."""
        fraction = min(self._sample_step / self.beta_anneal_steps, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def _compute_vlm_priority(self, env_info):
        """
        Compute VLM-based priority from env_info.

        Priority design:
        - Low progress (0-0.1): Low priority (ineffective exploration)
        - Mid progress (0.2-0.7): HIGH priority (most informative, "almost succeeded")
        - High progress (0.8-1.0): Medium priority (already learned well)

        This encourages sampling trajectories with highest information entropy.
        """
        # Extract VLM scores with defaults
        action_score = 0.5  # Neutral default
        progress = 0.0

        if env_info is not None:
            action_score = float(env_info.get(self.action_score_key, 0.5))
            progress = float(env_info.get(self.progress_key, 0.0))

        # Non-linear progress weighting: peak at mid-progress (most informative)
        # Using a bell-curve-like function centered at progress=0.5
        # progress_bonus = 4 * progress * (1 - progress)  # peaks at 0.5 with value 1.0

        # Alternative: boost mid-range, keep high progress valuable
        if progress < 0.1:
            progress_factor = progress  # Low priority for ineffective
        elif progress < 0.8:
            progress_factor = progress + 0.3  # Boost "almost succeeded" trajectories
        else:
            progress_factor = progress  # Success is still valuable

        # Clamp to [0, 1]
        progress_factor = min(1.0, progress_factor)

        # Weighted combination
        vlm_score = (
            self.action_score_weight * action_score +
            self.progress_weight * progress_factor
        )

        return vlm_score, action_score, progress

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        env_info=None,
        **kwargs
    ):
        """Add sample with hybrid VLM + TD priority.

        Samples with progress < min_progress_threshold are filtered out
        to keep buffer focused on informative trajectories.
        """
        # Compute VLM score
        vlm_score, action_score, progress = self._compute_vlm_priority(env_info)

        # Filter low-progress samples (keep buffer efficient)
        if progress < self.min_progress_threshold:
            self._filtered_count += 1
            return  # Skip this sample

        idx = self._top

        # Store VLM scores
        self._vlm_action_score[idx] = action_score
        self._vlm_progress[idx] = progress
        self._vlm_priority_score[idx] = vlm_score

        # Initialize TD error (will be updated after first training)
        self._td_errors[idx] = 1.0  # Start with max TD error (new samples are important)
        self._has_td_error[idx] = False

        # Compute initial priority based on mode
        priority = self._compute_priority(idx, vlm_score, td_error=1.0)
        self._priorities[idx] = priority

        self.max_priority = max(self.max_priority, priority)

        # Add to priority tree
        self.tree.add(priority, idx)

        # Call parent add_sample
        super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            env_info=env_info,
            **kwargs
        )

    def _compute_priority(self, idx, vlm_score, td_error):
        """
        Compute combined priority based on mode.

        Modes:
        - vlm_only: priority = vlm_score^α
        - td_only: priority = |td_error|^α
        - hybrid: priority = vlm_score^(α×w_vlm) × |td_error|^(α×w_td)
        """
        # Clip TD-error to prevent priority explosion
        clipped_td_error = min(abs(td_error), self.max_td_error)

        if self.priority_mode == 'vlm_only':
            priority = (vlm_score + self.epsilon) ** self.alpha

        elif self.priority_mode == 'td_only':
            priority = (clipped_td_error + self.epsilon) ** self.alpha

        else:  # hybrid mode
            vlm_priority = (vlm_score + self.epsilon) ** (self.alpha * self.vlm_weight)
            td_priority = (clipped_td_error + self.epsilon) ** (self.alpha * self.td_weight)
            priority = vlm_priority * td_priority

        return priority

    def random_batch(self, batch_size):
        """Sample batch with VLM-based priorities."""
        self._sample_step += 1

        if self._size < batch_size:
            # Not enough samples, use uniform sampling
            batch = super().random_batch(batch_size)
            # Add default weights
            batch['weights'] = np.ones(batch_size, dtype=np.float32)
            batch['indices'] = np.arange(batch_size)
            return batch

        # Prioritized sampling using sum tree
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            leaf_idx, priority, data_idx = self.tree.get_leaf(v)

            # Ensure valid index
            data_idx = int(data_idx) % self._size
            indices.append(data_idx)
            priorities.append(max(priority, self.epsilon))

        indices = np.array(indices)
        priorities = np.array(priorities)

        # Compute importance sampling weights
        sampling_probs = priorities / (self.tree.total_priority + 1e-8)
        weights = (self._size * sampling_probs + 1e-8) ** (-self.beta)
        # Clip IS weights to prevent extreme gradient updates (stability)
        weights = np.clip(weights, 0.0, self.max_is_weight)
        weights = weights / (weights.max() + 1e-8)  # Normalize

        # Build batch
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            weights=weights.astype(np.float32),
            indices=indices,
            # VLM scores for analysis
            vlm_action_score=self._vlm_action_score[indices],
            vlm_progress=self._vlm_progress[indices],
            vlm_priority_score=self._vlm_priority_score[indices],
            # TD and combined priority
            td_errors=self._td_errors[indices],
            priorities=self._priorities[indices],
        )

        # Add env_info
        for key in self._env_info_keys:
            batch[key] = self._env_infos[key][indices]

        return batch

    def update_priorities(self, indices, td_errors):
        """
        Update priorities with new TD-errors after training.

        This should be called after each training step to update
        priorities based on the latest TD-errors.

        Args:
            indices: Sample indices from the batch
            td_errors: TD-errors for each sample (numpy array)
        """
        for idx, td_error in zip(indices, td_errors):
            idx = int(idx)

            # Store TD error
            self._td_errors[idx] = abs(td_error)
            self._has_td_error[idx] = True

            # Get VLM score
            vlm_score = self._vlm_priority_score[idx]

            # Recompute priority
            priority = self._compute_priority(idx, vlm_score, td_error)
            self._priorities[idx] = priority

            self.max_priority = max(self.max_priority, priority)

            # Update tree
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

    def get_diagnostics(self):
        """Return buffer diagnostics including VLM and TD priority stats."""
        diagnostics = OrderedDict([
            ('size', self._size),
            ('vlm_priority/mode', self.priority_mode),
            ('vlm_priority/beta', self.beta),
            ('vlm_priority/max_priority', self.max_priority),
            ('vlm_priority/max_is_weight_limit', self.max_is_weight),
            ('vlm_priority/max_td_error_limit', self.max_td_error),
            ('vlm_priority/min_progress_threshold', self.min_progress_threshold),
            ('vlm_priority/filtered_count', self._filtered_count),
        ])

        if self._size > 0:
            # VLM statistics
            diagnostics.update({
                'vlm_priority/mean_priority_score': float(self._vlm_priority_score[:self._size].mean()),
                'vlm_priority/mean_action_score': float(self._vlm_action_score[:self._size].mean()),
                'vlm_priority/mean_progress': float(self._vlm_progress[:self._size].mean()),
            })

            # TD-error statistics (only for samples that have been trained)
            has_td = self._has_td_error[:self._size]
            if has_td.any():
                td_errors = self._td_errors[:self._size][has_td]
                clipped_count = (td_errors >= self.max_td_error).sum()
                diagnostics.update({
                    'vlm_priority/mean_td_error': float(td_errors.mean()),
                    'vlm_priority/max_td_error': float(td_errors.max()),
                    'vlm_priority/td_coverage': float(has_td.mean()),  # % samples with TD
                    'vlm_priority/td_clipped_ratio': float(clipped_count / len(td_errors)),
                })

            # Combined priority statistics
            diagnostics.update({
                'vlm_priority/mean_priority': float(self._priorities[:self._size].mean()),
                'vlm_priority/std_priority': float(self._priorities[:self._size].std()),
            })

        return diagnostics


class VLMPrioritizedEnvReplayBuffer(VLMPrioritizedReplayBuffer):
    """
    VLM-Prioritized replay buffer that wraps an environment.
    Drop-in replacement for EnvReplayBuffer with VLM prioritization.
    """

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        env_info_sizes=None,
        **kwargs
    ):
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            **kwargs
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            **kwargs
        )
