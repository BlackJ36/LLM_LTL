"""
Cleanup task propositions.

Task: Pick and place object into bin, then push another object to target.
LTL: ◇(pnp_grasped ∧ ◇(pnp_in_bin ∧ ◇(push_contact ∧ ◇push_complete)))
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite cleanup.py
TABLE_HEIGHT = 0.8
GRIPPER_CLOSED_THRESHOLD = 0.02
PUSH_COMPLETE_THRESHOLD = 0.1  # 10cm distance to target


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Cleanup task.

    Propositions:
    - pnp_grasped: Gripper is holding PNP object
    - pnp_in_bin: PNP object is in bin
    - push_contact: Gripper is near push object
    - push_complete: Push object is at target position

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check pnp_grasped
    if _check_pnp_grasped(obs, info):
        events.add('pnp_grasped')

    # Check pnp_in_bin
    if _check_pnp_in_bin(obs, info):
        events.add('pnp_in_bin')

    # Check push_contact
    if _check_push_contact(obs, info):
        events.add('push_contact')

    # Check push_complete
    if _check_push_complete(obs, info):
        events.add('push_complete')

    return events


def _check_pnp_grasped(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if gripper is holding PNP object."""
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos
        return gripper_opening < GRIPPER_CLOSED_THRESHOLD

    return info.get('grasp', False)


def _check_pnp_in_bin(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if PNP object is in bin."""
    # Use info flag
    if 'pnp_success' in info:
        return bool(info['pnp_success'])

    if 'objects_in_bins' in info:
        return np.any(info['objects_in_bins'])

    return False


def _check_push_contact(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if gripper is near push object (ready to push)."""
    # Get gripper position
    eef_pos = obs.get('robot0_eef_pos', None)
    if eef_pos is None:
        return False

    # Get push object position
    push_obj_pos = info.get('push_obj_pos', None)
    if push_obj_pos is None:
        return False

    dist = np.linalg.norm(np.array(eef_pos)[:2] - np.array(push_obj_pos)[:2])
    return dist < 0.1  # 10cm


def _check_push_complete(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if push object is at target position."""
    # Use info flag
    if 'push_success' in info:
        return bool(info['push_success'])

    # Check distance to target
    if 'push_dist_to_target' in info:
        return info['push_dist_to_target'] < PUSH_COMPLETE_THRESHOLD

    return info.get('success', False)
