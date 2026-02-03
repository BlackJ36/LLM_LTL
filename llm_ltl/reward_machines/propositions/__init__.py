"""
Task-specific proposition functions for Reward Machines.

Each module provides a get_events(obs, info) function that returns
a set of currently true propositions.
"""

from llm_ltl.reward_machines.propositions.base_props import (
    check_grasped,
    check_lifted,
    check_distance,
    check_contact,
)

__all__ = [
    'check_grasped',
    'check_lifted',
    'check_distance',
    'check_contact',
]
