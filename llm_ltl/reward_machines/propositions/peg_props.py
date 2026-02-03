"""
Peg Insertion task propositions.

Task: Insert peg into hole with proper alignment.
LTL: ◇(peg_grasped ∧ ◇(peg_aligned ∧ ◇(peg_positioned ∧ ◇peg_inserted)))
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite peg_in_hole.py
GRIPPER_CLOSED_THRESHOLD = 0.02
ALIGN_COS_THRESHOLD = 0.95  # cos > 0.95 means angle < 18 degrees
PERP_DIST_THRESHOLD = 0.05  # 5cm perpendicular distance
PARALLEL_OFFSET_THRESHOLD = 0.05  # 5cm parallel offset


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Peg Insertion task.

    Propositions:
    - peg_grasped: Gripper is holding peg
    - peg_aligned: Peg orientation is aligned with hole (cos > 0.95)
    - peg_positioned: Peg is close to hole axis (perpendicular distance < 5cm)
    - peg_inserted: Peg is inserted into hole (parallel offset in range)

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check peg_grasped
    if _check_peg_grasped(obs, info):
        events.add('peg_grasped')

    # Get alignment metrics
    cos_angle = info.get('angle', info.get('cos_angle', 0.0))
    perp_dist = info.get('d', info.get('perp_dist', float('inf')))
    parallel_offset = info.get('t', info.get('parallel_offset', float('inf')))

    # Check peg_aligned
    if _check_peg_aligned(cos_angle):
        events.add('peg_aligned')

    # Check peg_positioned
    if _check_peg_positioned(perp_dist):
        events.add('peg_positioned')

    # Check peg_inserted
    if _check_peg_inserted(obs, info, cos_angle, perp_dist, parallel_offset):
        events.add('peg_inserted')

    return events


def _check_peg_grasped(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if gripper is holding peg."""
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos
        return gripper_opening < GRIPPER_CLOSED_THRESHOLD

    return info.get('grasp', False)


def _check_peg_aligned(cos_angle: float) -> bool:
    """Check if peg orientation is aligned with hole."""
    return cos_angle > ALIGN_COS_THRESHOLD


def _check_peg_positioned(perp_dist: float) -> bool:
    """Check if peg is close to hole axis."""
    return perp_dist < PERP_DIST_THRESHOLD


def _check_peg_inserted(obs: Dict[str, Any], info: Dict[str, Any],
                         cos_angle: float, perp_dist: float,
                         parallel_offset: float) -> bool:
    """Check if peg is inserted into hole."""
    # Use success flag first
    if info.get('success', False):
        return True

    # All conditions must be met
    if cos_angle <= ALIGN_COS_THRESHOLD:
        return False
    if perp_dist >= PERP_DIST_THRESHOLD:
        return False
    if abs(parallel_offset) >= PARALLEL_OFFSET_THRESHOLD:
        return False

    return True
