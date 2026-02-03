"""
VLM Reward Wrapper for robosuite environments.

Wraps a robosuite environment to add VLM-based reward signals
for guiding action primitive selection.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from .vlm_client import QwenVLClient


class VLMRewardWrapper:
    """
    Wrapper that adds VLM-based reward to a robosuite environment.

    The VLM evaluates:
    1. Binary judgment: Is the selected action primitive reasonable?
    2. Progress evaluation: How close is the current state to task completion?

    These are combined into an auxiliary reward signal added to the environment reward.
    """

    def __init__(
        self,
        env,
        task_description: str,
        skill_controller,
        vlm_client: Optional[QwenVLClient] = None,
        vlm_reward_scale: float = 0.5,
        action_weight: float = 0.2,   # Lower: MAPLE's dual-policy already learns action selection
        progress_weight: float = 0.8,  # Higher: encourages sampling of effective trajectories
        eval_frequency: int = 1,
        camera_name: str = "frontview",
        camera_height: int = 256,
        camera_width: int = 256,
        enabled: bool = True,
        warmup_steps: int = 0,
        image_history_size: int = 4,
        # Plan B: Potential-based shaping
        use_potential_shaping: bool = True,
        gamma: float = 0.99,
        # Plan D: Curriculum learning
        curriculum_mode: str = "none",  # "none", "linear", "step", "cosine"
        curriculum_start: float = 1.0,
        curriculum_end: float = 0.1,
    ):
        """
        Initialize the VLM reward wrapper.

        VLM provides two signals:
        1. Action score: How appropriate is the selected primitive (0-1)
           - Lower weight recommended: MAPLE's dual-policy already learns action selection
        2. Progress: How close to task completion (0-1)
           - Higher weight recommended: encourages sampling of effective trajectories
           - Complements env's staged_rewards with visual perspective

        Design rationale:
        - Environment already provides staged milestone rewards (reach → grasp → lift)
        - VLM action_score is auxiliary (dual-policy handles primitive selection)
        - VLM progress is valuable for prioritized replay sampling

        Args:
            env: The robosuite environment to wrap
            task_description: Natural language description of the task
            skill_controller: MAPLE skill controller for primitive name lookup
            vlm_client: Optional pre-configured VLM client
            vlm_reward_scale: Scale factor for VLM reward
            action_weight: Weight for action score reward
            progress_weight: Weight for progress-based reward
            eval_frequency: Evaluate VLM every N steps (1 = every step)
            camera_name: Camera to use for rendering
            camera_height: Height of rendered image
            camera_width: Width of rendered image
            enabled: Whether VLM reward is active
            warmup_steps: Number of steps before enabling VLM
            image_history_size: Number of historical images to keep
            use_potential_shaping: Use potential-based progress shaping (allows negative reward)
            gamma: Discount factor for potential shaping
            curriculum_mode: "none", "linear", "step", or "cosine"
            curriculum_start: Initial VLM weight (default 1.0)
            curriculum_end: Final VLM weight (default 0.1)
        """
        self.env = env
        self.task_description = task_description
        self.skill_controller = skill_controller
        self.vlm_reward_scale = vlm_reward_scale
        self.action_weight = action_weight
        self.progress_weight = progress_weight
        self.eval_frequency = eval_frequency
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.image_history_size = image_history_size

        # Potential-based shaping parameters
        self.use_potential_shaping = use_potential_shaping
        self.gamma = gamma

        # Progress smoothing (to handle VLM evaluation instability)
        self.progress_smoothing_alpha = 0.3  # EMA alpha: higher = more responsive, lower = smoother
        self.progress_negative_reward_limit = -0.1  # Limit negative progress reward

        # Curriculum learning parameters
        self.curriculum_mode = curriculum_mode
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end
        self._current_epoch = 0
        self._total_epochs = 500  # Default, will be updated by trainer

        # Initialize VLM client
        self.vlm_client = vlm_client or QwenVLClient()

        # State tracking
        self._step_count = 0
        self._total_steps = 0  # Global step counter across all episodes
        self._prev_progress = None
        self._smoothed_progress = None  # EMA-smoothed progress value
        self._max_progress = 0.0  # Track max progress in episode
        self._last_vlm_info = {}
        self._warmup_complete = (warmup_steps == 0)

        # Image history buffer (stores last N images, oldest first)
        self._image_history: list[np.ndarray] = []

        # Trajectory-level VLM statistics
        self._traj_vlm_rewards = []       # All VLM rewards in trajectory
        self._traj_action_scores = []     # All action scores (for sampling)
        self._traj_progress_values = []   # All progress values (for sampling)

        # Reward spreading: distribute VLM reward across eval_frequency steps
        self._cached_vlm_reward = 0.0     # Cached reward per step (spread)
        self._cached_vlm_info = {}        # Cached VLM info for non-eval steps
        self._steps_since_eval = 0        # Steps since last VLM evaluation

        # Default VLM info keys (for consistent env_info structure)
        self._default_vlm_info = {
            "vlm_action_score": 0.0,      # Soft score for selected action (0-1)
            "vlm_reasonable": 0.0,        # Derived: action_score > 0.5 (for sampling)
            "vlm_progress": 0.0,          # Task progress (0-1, for sampling)
            "vlm_action_reward": 0.0,     # Action component of reward
            "vlm_progress_reward": 0.0,   # Progress component of reward
            "vlm_total_reward": 0.0,      # Combined VLM reward
        }

        # Get available primitives from skill controller
        self._available_primitives = self._get_available_primitives()

    def _get_available_primitives(self) -> list[str]:
        """Extract available primitive names from skill controller."""
        if hasattr(self.skill_controller, 'skill_names'):
            return list(self.skill_controller.skill_names)
        elif hasattr(self.skill_controller, 'skills'):
            return list(self.skill_controller.skills.keys())
        else:
            # Fallback: try to get from action space
            return [f"primitive_{i}" for i in range(self.env.action_skill_dim)]

    def _get_primitive_name(self, action: np.ndarray) -> str:
        """Get the name of the selected primitive from action vector."""
        if hasattr(self.skill_controller, 'get_skill_name_from_action'):
            return self.skill_controller.get_skill_name_from_action(action)
        else:
            # Fallback: decode from one-hot portion of action
            skill_dim = getattr(self.env, 'action_skill_dim', 0)
            if skill_dim > 0:
                skill_idx = np.argmax(action[:skill_dim])
                if skill_idx < len(self._available_primitives):
                    return self._available_primitives[skill_idx]
            return "unknown"

    def _render_image(self) -> np.ndarray:
        """Render the current scene from the specified camera."""
        # Use mujoco_py's sim.render for offscreen rendering
        image = self.env.sim.render(
            width=self.camera_width,
            height=self.camera_height,
            camera_name=self.camera_name,
        )
        # mujoco renders upside down, flip it
        image = np.flipud(image)
        return image

    def reset(self, **kwargs):
        """Reset the environment and VLM state."""
        obs = self.env.reset(**kwargs)
        self._step_count = 0
        self._prev_progress = None
        self._smoothed_progress = None  # Reset smoothed progress
        self._max_progress = 0.0  # Reset max progress
        self._last_vlm_info = {}
        self._image_history = []  # Clear image history on reset

        # Reset trajectory-level statistics
        self._traj_vlm_rewards = []
        self._traj_action_scores = []
        self._traj_progress_values = []

        # Reset reward spreading state
        self._cached_vlm_reward = 0.0
        self._cached_vlm_info = {}
        self._steps_since_eval = 0
        return obs

    def step(self, action: np.ndarray, **kwargs):
        """
        Step the environment and add VLM reward.

        VLM provides:
        - action_score: Soft score for selected primitive (0-1)
        - progress: Task completion progress (0-1, smoothed)

        These are combined into reward and also exposed for sampling strategy.

        Args:
            action: Action to execute (includes primitive selection + parameters)
            **kwargs: Additional arguments passed to underlying environment (e.g., image_obs_in_info)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Execute action in environment
        obs, env_reward, done, info = self.env.step(action, **kwargs)
        self._step_count += 1
        self._total_steps += 1

        # Check warmup completion
        if not self._warmup_complete and self._total_steps >= self.warmup_steps:
            self._warmup_complete = True
            print(f"[VLM] Warmup complete after {self._total_steps} steps. VLM reward now active.")

        # Initialize VLM reward and info
        vlm_reward = 0.0
        vlm_info = self._default_vlm_info.copy()
        self._steps_since_eval += 1

        # Evaluate with VLM if enabled, warmup complete, and at eval frequency
        should_evaluate = (self.enabled and self._warmup_complete and
                          (self._step_count % self.eval_frequency == 0))

        if should_evaluate:
            try:
                # Get current scene image
                image = self._render_image()

                # Get selected primitive name
                primitive_name = self._get_primitive_name(action)
                image_history = self._image_history.copy() if self._image_history else None

                # === Action Scoring (soft scores for all primitives) ===
                scores = self.vlm_client.score_all_primitives(
                    image=image,
                    task_description=self.task_description,
                    available_primitives=self._available_primitives,
                    image_history=image_history,
                )
                action_score = scores.get(primitive_name, 0.5)
                is_reasonable = action_score > 0.5  # Derived for sampling strategy

                # Action reward: directly use the soft score (0-1)
                action_reward = action_score

                # === Progress Evaluation ===
                raw_progress, stage, next_step = self.vlm_client.evaluate_progress(
                    image=image,
                    task_description=self.task_description,
                    image_history=image_history,
                )

                # Progress smoothing (handle VLM instability)
                if self._smoothed_progress is None:
                    smoothed_progress = raw_progress
                else:
                    smoothed_progress = (self.progress_smoothing_alpha * raw_progress +
                                        (1 - self.progress_smoothing_alpha) * self._smoothed_progress)

                # Track max progress
                self._max_progress = max(self._max_progress, smoothed_progress)

                # === Progress Reward ===
                if self._prev_progress is not None:
                    if self.use_potential_shaping:
                        # Potential shaping: r = γΦ(s') - Φ(s)
                        progress_reward = self.gamma * smoothed_progress - self._prev_progress
                        progress_reward = max(self.progress_negative_reward_limit, progress_reward)
                    else:
                        # Simple delta (positive only)
                        progress_reward = max(0, smoothed_progress - self._prev_progress)
                else:
                    progress_reward = 0.0  # No reward on first step

                # Combined VLM reward for this evaluation period
                total_vlm_reward = self.action_weight * action_reward + self.progress_weight * progress_reward

                # === Reward Spreading ===
                # Spread the VLM reward across eval_frequency steps
                # This ensures consistent reward signal even with sparse VLM evaluation
                self._cached_vlm_reward = total_vlm_reward / self.eval_frequency
                vlm_reward = self._cached_vlm_reward
                self._steps_since_eval = 1  # Reset counter

                # Update image history
                self._image_history.append(image)
                if len(self._image_history) > self.image_history_size:
                    self._image_history.pop(0)

                # Update progress tracking
                self._prev_progress = smoothed_progress
                self._smoothed_progress = smoothed_progress

                # Build VLM info (for logging and sampling strategy)
                new_vlm_info = {
                    "vlm_action_score": action_score,
                    "vlm_reasonable": is_reasonable,  # For sampling strategy
                    "vlm_progress": smoothed_progress,  # For sampling strategy
                    "vlm_progress_raw": raw_progress,
                    "vlm_progress_max": self._max_progress,
                    "vlm_action_reward": action_reward,
                    "vlm_progress_reward": progress_reward,
                    "vlm_total_reward": total_vlm_reward,
                    "vlm_reward_per_step": self._cached_vlm_reward,  # Spread reward
                    "vlm_all_scores": scores,  # All primitive scores for debugging
                }
                self._last_vlm_info = new_vlm_info

                # Cache VLM info for non-eval steps
                self._cached_vlm_info = {
                    "vlm_action_score": action_score,
                    "vlm_reasonable": float(is_reasonable),
                    "vlm_progress": smoothed_progress,
                    "vlm_action_reward": action_reward,
                    "vlm_progress_reward": progress_reward,
                    "vlm_total_reward": total_vlm_reward,
                }

                # Numeric fields for env_info
                vlm_info = self._cached_vlm_info.copy()

                # Collect trajectory statistics (for sampling strategy)
                self._traj_vlm_rewards.append(total_vlm_reward)
                self._traj_action_scores.append(action_score)
                self._traj_progress_values.append(smoothed_progress)

            except Exception as e:
                # On error, use cached values if available
                vlm_reward = self._cached_vlm_reward
                if self._cached_vlm_info:
                    vlm_info = self._cached_vlm_info.copy()

        elif self.enabled and self._warmup_complete:
            # Non-evaluation step: use cached spread reward and info
            vlm_reward = self._cached_vlm_reward
            if self._cached_vlm_info:
                vlm_info = self._cached_vlm_info.copy()

        # Always add VLM info to step info
        info.update(vlm_info)

        # Add trajectory-level statistics when episode ends
        if done:
            info.update(self._compute_trajectory_stats())

        # Apply curriculum weight and scale
        curriculum_weight = self._get_curriculum_weight()
        vlm_reward_scaled = self.vlm_reward_scale * vlm_reward * curriculum_weight

        # Combine rewards
        total_reward = env_reward + vlm_reward_scaled

        # Store reward components in info
        info["env_reward"] = env_reward
        info["vlm_reward_scaled"] = vlm_reward_scaled
        info["vlm_curriculum_weight"] = curriculum_weight
        info["total_reward"] = total_reward

        return obs, total_reward, done, info

    def _compute_trajectory_stats(self) -> dict:
        """
        Compute trajectory-level VLM statistics.

        These stats are useful for:
        - Logging and debugging
        - Sampling strategy (prioritized replay)
        """
        stats = {}

        if len(self._traj_vlm_rewards) > 0:
            rewards = np.array(self._traj_vlm_rewards)
            action_scores = np.array(self._traj_action_scores)
            progress = np.array(self._traj_progress_values)

            # Reward statistics
            stats["vlm_traj_reward_sum"] = float(np.sum(rewards))
            stats["vlm_traj_reward_mean"] = float(np.mean(rewards))

            # Action score statistics (for sampling)
            stats["vlm_traj_action_score_mean"] = float(np.mean(action_scores))
            stats["vlm_traj_reasonable_rate"] = float(np.mean(action_scores > 0.5))

            # Progress statistics (for sampling)
            stats["vlm_traj_progress_final"] = float(progress[-1]) if len(progress) > 0 else 0.0
            stats["vlm_traj_progress_max"] = float(np.max(progress))
            stats["vlm_traj_progress_mean"] = float(np.mean(progress))

            # Progress improvement
            if len(progress) > 1:
                stats["vlm_traj_progress_delta"] = float(progress[-1] - progress[0])
            else:
                stats["vlm_traj_progress_delta"] = 0.0

            stats["vlm_traj_eval_count"] = float(len(self._traj_vlm_rewards))
        else:
            # No VLM evaluations
            stats["vlm_traj_reward_sum"] = 0.0
            stats["vlm_traj_reward_mean"] = 0.0
            stats["vlm_traj_action_score_mean"] = 0.0
            stats["vlm_traj_reasonable_rate"] = 0.0
            stats["vlm_traj_progress_final"] = 0.0
            stats["vlm_traj_progress_max"] = 0.0
            stats["vlm_traj_progress_mean"] = 0.0
            stats["vlm_traj_progress_delta"] = 0.0
            stats["vlm_traj_eval_count"] = 0.0

        return stats

    def set_task_description(self, task_description: str):
        """Update the task description for VLM evaluation."""
        self.task_description = task_description

    def set_enabled(self, enabled: bool):
        """Enable or disable VLM reward."""
        self.enabled = enabled

    def get_last_vlm_info(self) -> dict:
        """Get the info from the last VLM evaluation."""
        return self._last_vlm_info

    # Plan D: Curriculum learning methods
    def set_curriculum_epoch(self, epoch: int, total_epochs: int):
        """
        Update curriculum state. Called by trainer at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of training epochs
        """
        self._current_epoch = epoch
        self._total_epochs = max(total_epochs, 1)

    def _get_curriculum_weight(self) -> float:
        """
        Calculate VLM reward weight based on curriculum schedule.

        Returns:
            Weight multiplier for VLM reward (curriculum_start -> curriculum_end)
        """
        if self.curriculum_mode == "none":
            return 1.0

        progress = self._current_epoch / self._total_epochs

        if self.curriculum_mode == "linear":
            # Linear decay: start -> end
            return self.curriculum_start + (self.curriculum_end - self.curriculum_start) * progress

        elif self.curriculum_mode == "step":
            # Step decay: high weight for first 1/3, low weight for rest
            return self.curriculum_start if progress < 0.33 else self.curriculum_end

        elif self.curriculum_mode == "cosine":
            # Cosine annealing: smooth decay with slower start/end
            import math
            return self.curriculum_end + (self.curriculum_start - self.curriculum_end) * \
                   0.5 * (1 + math.cos(math.pi * progress))

        else:
            return 1.0

    # Delegate all other attributes to the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def action_dim(self):
        return self.env.action_dim

    @property
    def action_skill_dim(self):
        return getattr(self.env, 'action_skill_dim', 0)


class VLMRewardWrapperAsync(VLMRewardWrapper):
    """
    Async version of VLM reward wrapper.

    Evaluates VLM in a separate thread to avoid blocking training.
    The VLM reward is applied with a delay (from the previous evaluation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import queue
        import threading

        self._eval_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._pending_vlm_reward = 0.0
        self._pending_vlm_info = {}
        self._worker_thread = None
        self._stop_worker = False

    def _worker(self):
        """Background worker for VLM evaluation."""
        while not self._stop_worker:
            try:
                request = self._eval_queue.get(timeout=0.1)
                if request is None:
                    continue

                image, primitive_name, image_history = request

                # Action scoring
                scores = self.vlm_client.score_all_primitives(
                    image=image,
                    task_description=self.task_description,
                    available_primitives=self._available_primitives,
                    image_history=image_history,
                )
                action_score = scores.get(primitive_name, 0.5)
                action_reward = action_score

                # Progress evaluation
                raw_progress, stage, next_step = self.vlm_client.evaluate_progress(
                    image=image,
                    task_description=self.task_description,
                    image_history=image_history,
                )

                # Progress smoothing
                if self._smoothed_progress is None:
                    smoothed_progress = raw_progress
                else:
                    smoothed_progress = (self.progress_smoothing_alpha * raw_progress +
                                        (1 - self.progress_smoothing_alpha) * self._smoothed_progress)

                # Progress reward
                if self._prev_progress is not None:
                    if self.use_potential_shaping:
                        progress_reward = self.gamma * smoothed_progress - self._prev_progress
                        progress_reward = max(self.progress_negative_reward_limit, progress_reward)
                    else:
                        progress_reward = max(0, smoothed_progress - self._prev_progress)
                else:
                    progress_reward = 0.0

                # Combined reward
                vlm_reward = self.action_weight * action_reward + self.progress_weight * progress_reward

                # Update state
                self._prev_progress = smoothed_progress
                self._smoothed_progress = smoothed_progress
                self._max_progress = max(self._max_progress, smoothed_progress)

                vlm_info = {
                    "vlm_action_score": action_score,
                    "vlm_reasonable": action_score > 0.5,
                    "vlm_progress": smoothed_progress,
                    "vlm_action_reward": action_reward,
                    "vlm_progress_reward": progress_reward,
                    "vlm_total_reward": vlm_reward,
                }

                try:
                    self._result_queue.put_nowait((vlm_reward, vlm_info))
                except:
                    pass

            except:
                continue

    def _start_worker(self):
        """Start the background evaluation worker."""
        import threading
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_worker = False
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    def _stop_worker_thread(self):
        """Stop the background worker."""
        self._stop_worker = True
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)

    def reset(self, **kwargs):
        """Reset and start worker."""
        self._start_worker()
        return super().reset(**kwargs)

    def step(self, action: np.ndarray, **kwargs):
        """Step with async VLM evaluation and reward spreading."""
        obs, env_reward, done, info = self.env.step(action, **kwargs)
        self._step_count += 1
        self._total_steps += 1

        if not self._warmup_complete and self._total_steps >= self.warmup_steps:
            self._warmup_complete = True
            print(f"[VLM] Warmup complete after {self._total_steps} steps.")

        vlm_info = self._default_vlm_info.copy()

        # Check for completed evaluation from background worker
        if self._warmup_complete:
            try:
                vlm_reward, new_vlm_info = self._result_queue.get_nowait()
                # Spread reward across eval_frequency steps
                self._pending_vlm_reward = vlm_reward / self.eval_frequency
                self._pending_vlm_info = new_vlm_info
                self._last_vlm_info = new_vlm_info

                # Cache VLM info for all steps in this evaluation period
                self._cached_vlm_info = {
                    "vlm_action_score": new_vlm_info.get("vlm_action_score", 0.0),
                    "vlm_reasonable": float(new_vlm_info.get("vlm_reasonable", False)),
                    "vlm_progress": new_vlm_info.get("vlm_progress", 0.0),
                    "vlm_action_reward": new_vlm_info.get("vlm_action_reward", 0.0),
                    "vlm_progress_reward": new_vlm_info.get("vlm_progress_reward", 0.0),
                    "vlm_total_reward": new_vlm_info.get("vlm_total_reward", 0.0),
                }

                self._traj_vlm_rewards.append(vlm_reward)
                self._traj_action_scores.append(new_vlm_info.get("vlm_action_score", 0.0))
                self._traj_progress_values.append(new_vlm_info.get("vlm_progress", 0.0))
            except:
                pass

        # Use cached VLM info for all steps
        if self._cached_vlm_info:
            vlm_info = self._cached_vlm_info.copy()

        info.update(vlm_info)

        # Submit new evaluation request at eval_frequency
        if self.enabled and self._warmup_complete and (self._step_count % self.eval_frequency == 0):
            try:
                image = self._render_image()
                primitive_name = self._get_primitive_name(action)
                try:
                    history_copy = self._image_history.copy() if self._image_history else None
                    self._eval_queue.put_nowait((image, primitive_name, history_copy))
                    self._image_history.append(image)
                    if len(self._image_history) > self.image_history_size:
                        self._image_history.pop(0)
                except:
                    pass
            except:
                pass

        # Apply curriculum weight and scale (using spread reward)
        curriculum_weight = self._get_curriculum_weight()
        vlm_reward_scaled = self.vlm_reward_scale * self._pending_vlm_reward * curriculum_weight

        total_reward = env_reward + vlm_reward_scaled

        info["env_reward"] = env_reward
        info["vlm_reward_scaled"] = vlm_reward_scaled
        info["vlm_curriculum_weight"] = curriculum_weight
        info["total_reward"] = total_reward

        if done:
            info.update(self._compute_trajectory_stats())

        return obs, total_reward, done, info

    def close(self):
        """Clean up worker thread."""
        self._stop_worker_thread()
        if hasattr(self.env, 'close'):
            self.env.close()
