"""
VLM-aware Batch RL Algorithm.

Extends TorchBatchRLAlgorithm to support priority updates
after each training step for VLM-prioritized replay.
"""

import gtimer as gt

from maple.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from maple.data_management.vlm_prioritized_replay_buffer import VLMPrioritizedReplayBuffer


class VLMBatchRLAlgorithm(TorchBatchRLAlgorithm):
    """
    Batch RL algorithm with VLM priority update support.

    After each training step, updates replay buffer priorities
    based on TD-errors from the trainer.
    """

    def __init__(
        self,
        *args,
        update_priorities=True,
        # Early stopping parameters
        early_stop_success_threshold=0.95,  # Stop if success rate >= this
        early_stop_patience=3,               # Number of consecutive evals above threshold
        early_stop_enabled=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.update_priorities = update_priorities

        # Early stopping
        self.early_stop_success_threshold = early_stop_success_threshold
        self.early_stop_patience = early_stop_patience
        self.early_stop_enabled = early_stop_enabled
        self._consecutive_success_count = 0
        self._early_stopped = False

        # Check if replay buffer supports priority updates
        self._supports_priority_update = isinstance(
            self.replay_buffer,
            VLMPrioritizedReplayBuffer
        )

    def _train(self):
        """Training loop with priority updates and curriculum learning."""
        if self.min_num_steps_before_training > 0 and not self._eval_only:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=True,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs + 1),
                save_itrs=True,
        ):
            # Plan D: Update curriculum epoch for VLM reward wrapper
            self._update_curriculum_epoch(epoch)

            for pre_epoch_func in self.pre_epoch_funcs:
                pre_epoch_func(self, epoch)

            if epoch % self._eval_epoch_freq == 0:
                eval_paths = self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )

                # Early stopping check
                if self.early_stop_enabled and self._check_early_stop(eval_paths):
                    print(f"[Early Stop] Success rate >= {self.early_stop_success_threshold} "
                          f"for {self.early_stop_patience} consecutive evals. Stopping at epoch {epoch}.")
                    self._early_stopped = True
                    self._end_epoch(epoch)
                    break

            gt.stamp('evaluation sampling')

            if not self._eval_only:
                for _ in range(self.num_train_loops_per_epoch):
                    if epoch % self._expl_epoch_freq == 0:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=True,
                        )
                        gt.stamp('exploration sampling', unique=False)

                        self.replay_buffer.add_paths(new_expl_paths)
                        gt.stamp('data storing', unique=False)

                    if not self._no_training:
                        self.training_mode(True)
                        for _ in range(self.num_trains_per_train_loop):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size)
                            self.trainer.train(train_data)

                            # Update priorities after training
                            self._update_priorities(train_data)

                        gt.stamp('training', unique=False)
                        self.training_mode(False)

            self._end_epoch(epoch)

    def _update_priorities(self, train_data):
        """Update replay buffer priorities based on TD-errors."""
        if not self.update_priorities:
            return

        if not self._supports_priority_update:
            return

        # Get TD errors from trainer
        td_errors = None
        if hasattr(self.trainer, 'get_td_errors'):
            td_errors = self.trainer.get_td_errors()

        if td_errors is None:
            return

        # Get indices from training data
        indices = train_data.get('indices', None)
        if indices is None:
            return

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

    def _check_early_stop(self, eval_paths) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            eval_paths: Evaluation paths from current epoch

        Returns:
            True if training should stop
        """
        if not eval_paths:
            return False

        # Calculate success rate from eval paths
        successes = []
        for path in eval_paths:
            env_infos = path.get('env_infos', {})
            # Check final success
            if 'success' in env_infos:
                final_success = env_infos['success'][-1] if len(env_infos['success']) > 0 else 0
                successes.append(final_success)

        if not successes:
            return False

        success_rate = sum(successes) / len(successes)

        if success_rate >= self.early_stop_success_threshold:
            self._consecutive_success_count += 1
            print(f"[Early Stop] Success rate {success_rate:.2%} >= {self.early_stop_success_threshold:.0%} "
                  f"({self._consecutive_success_count}/{self.early_stop_patience})")
        else:
            self._consecutive_success_count = 0

        return self._consecutive_success_count >= self.early_stop_patience

    def _update_curriculum_epoch(self, epoch: int):
        """
        Update curriculum epoch for VLM reward wrappers (Plan D).

        Calls set_curriculum_epoch on exploration and evaluation environments
        if they support it.
        """
        total_epochs = self.num_epochs

        # Update exploration env
        env = self.expl_env
        while env is not None:
            if hasattr(env, 'set_curriculum_epoch'):
                env.set_curriculum_epoch(epoch, total_epochs)
                break
            # Unwrap if needed
            env = getattr(env, 'env', None) or getattr(env, '_wrapped_env', None)

        # Update evaluation env
        env = self.eval_env
        while env is not None:
            if hasattr(env, 'set_curriculum_epoch'):
                env.set_curriculum_epoch(epoch, total_epochs)
                break
            env = getattr(env, 'env', None) or getattr(env, '_wrapped_env', None)
