"""
SAC Hybrid Trainer with VLM-based Importance Sampling.

Extends SACHybridTrainer to support importance sampling weights
from VLM-prioritized experience replay.
"""

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

import maple.torch.pytorch_util as ptu
from maple.core.eval_util import create_stats_ordered_dict
from maple.torch.sac.sac_hybrid import SACHybridTrainer, SACLosses, LossStatistics


class SACHybridVLMTrainer(SACHybridTrainer):
    """
    SAC Hybrid Trainer with VLM importance sampling support.

    When use_importance_sampling=True, the Q-function losses are weighted
    by importance sampling weights from prioritized replay.
    """

    def __init__(
        self,
        *args,
        use_importance_sampling=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_importance_sampling = use_importance_sampling

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        """
        Compute losses with optional importance sampling weights.
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Get importance sampling weights if available
        if self.use_importance_sampling and 'weights' in batch:
            weights = batch['weights']
            # Handle both numpy and tensor inputs
            if isinstance(weights, np.ndarray):
                is_weights = ptu.from_numpy(weights).unsqueeze(1)
            else:
                is_weights = weights.unsqueeze(1) if weights.dim() == 1 else weights
        else:
            is_weights = None

        """
        Policy and Alpha Loss
        """
        dd = self.get_dist_dict(obs, one_hot=self.one_hot)
        log_pi_s = dd.get('log_pi_s', None)
        log_pi_p = dd['log_pi_p']

        if self.use_automatic_entropy_tuning:
            if log_pi_s is not None:
                alpha_s_loss = -self.log_alpha_s * self.reduce_tensor(
                    log_pi_s + self.target_entropy_s, dd
                ).detach()
                alpha_s = self.log_alpha_s.exp()
            else:
                alpha_s_loss = None
                alpha_s = None

            alpha_p_loss = -self.log_alpha_p * self.reduce_tensor(
                log_pi_p + self.target_entropy_p, dd
            ).detach()
            alpha_p = self.log_alpha_p.exp()
        else:
            if log_pi_s is not None:
                alpha_s_loss = 0
                alpha_s = 1
            else:
                alpha_s_loss = None
                alpha_s = None

            alpha_p_loss = 0
            alpha_p = 1

        q_new_actions = torch.min(
            self.qf1(dd['obs'], dd['actions']),
            self.qf2(dd['obs'], dd['actions']),
        )
        alpha_log_pi = alpha_p * log_pi_p
        if log_pi_s is not None:
            alpha_log_pi += (alpha_s * log_pi_s)

        policy_loss = alpha_log_pi - q_new_actions
        policy_loss = self.reduce_tensor(policy_loss, dd)

        """
        QF Loss (with importance sampling)
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist_dict = self.get_dist_dict(next_obs)
        new_next_actions = next_dist_dict['actions']
        new_log_pi_s = next_dist_dict.get('log_pi_s', None)
        new_log_pi_p = next_dist_dict['log_pi_p']
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        )
        target_q_values = target_q_values - (alpha_p * new_log_pi_p)
        if new_log_pi_s is not None:
            target_q_values -= (alpha_s * new_log_pi_s)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        # Compute element-wise TD errors
        td_error1 = q1_pred - q_target.detach()
        td_error2 = q2_pred - q_target.detach()

        # Apply importance sampling weights if available
        if is_weights is not None:
            # Weighted MSE loss
            qf1_loss = (is_weights * (td_error1 ** 2)).mean()
            qf2_loss = (is_weights * (td_error2 ** 2)).mean()
        else:
            # Standard MSE loss
            qf1_loss = (td_error1 ** 2).mean()
            qf2_loss = (td_error2 ** 2).mean()

        # Store TD errors for priority updates
        self._last_td_errors = (td_error1.detach() + td_error2.detach()) / 2

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            dd = self.get_dist_dict(obs)

            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis S',
                ptu.get_numpy(dd.get('log_pi_s', torch.zeros(1))),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis P',
                ptu.get_numpy(dd['log_pi_p']),
            ))
            # Get policy statistics if available
            if hasattr(self.policy, 'policy') and hasattr(self.policy.policy, 'policy1'):
                policy1 = self.policy.policy.policy1
                if hasattr(policy1, 'get_log_p_statistics'):
                    policy_s_statistics = policy1.get_log_p_statistics(obs)
                    for k, v in policy_s_statistics.items():
                        eval_statistics['policy/'+k] = v
            if self.use_automatic_entropy_tuning:
                if alpha_s is not None:
                    eval_statistics['Alpha S'] = alpha_s.item()
                    eval_statistics['Alpha S Loss'] = alpha_s_loss.item()
                eval_statistics['Alpha P'] = alpha_p.item()
                eval_statistics['Alpha P Loss'] = alpha_p_loss.item()

            # Add importance sampling statistics
            if is_weights is not None:
                eval_statistics['IS/weights_mean'] = float(is_weights.mean())
                eval_statistics['IS/weights_std'] = float(is_weights.std())
                eval_statistics['IS/weights_min'] = float(is_weights.min())
                eval_statistics['IS/weights_max'] = float(is_weights.max())

            # Add VLM priority statistics if available
            if 'vlm_priority' in batch:
                eval_statistics['VLM/batch_priority_mean'] = float(batch['vlm_priority'].mean())
                eval_statistics['VLM/batch_reasonable_mean'] = float(batch['vlm_reasonable'].mean())

        return SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_s_loss=alpha_s_loss,
            alpha_p_loss=alpha_p_loss,
        ), eval_statistics

    def get_td_errors(self):
        """Get TD errors from last training step for priority updates."""
        if hasattr(self, '_last_td_errors'):
            return ptu.get_numpy(self._last_td_errors).flatten()
        return None

    def get_batch_indices(self):
        """Get indices from last training batch."""
        if hasattr(self, '_last_batch_indices'):
            return self._last_batch_indices
        return None

    def train_from_torch(self, batch):
        """Train with priority update support."""
        # Store batch indices for priority updates
        if 'indices' in batch:
            self._last_batch_indices = batch['indices']

        # Call parent training
        super().train_from_torch(batch)
