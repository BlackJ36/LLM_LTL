"""
Door task propositions.

Task: Open a door by rotating the handle and pulling.
LTL: ◇(handle_reached ∧ ◇door_opened)
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite door.py
HANDLE_REACH_THRESHOLD = 0.05  # 5cm
DOOR_OPEN_THRESHOLD = 0.3  # 0.3 radians (~17 degrees)


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Door task.

    Propositions:
    - handle_reached: Gripper is close to door handle
    - door_opened: Door hinge angle > threshold

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check handle_reached
    if _check_handle_reached(obs, info):
        events.add('handle_reached')

    # Check door_opened
    if _check_door_opened(obs, info):
        events.add('door_opened')

    return events


def _check_handle_reached(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if gripper is close to handle."""
    # Try info for gripper-to-handle distance
    grip_to_handle = info.get('gripper_to_handle', None)
    if grip_to_handle is not None:
        dist = np.linalg.norm(np.array(grip_to_handle))
        return dist < HANDLE_REACH_THRESHOLD

    # Compute from positions
    handle_pos = info.get('handle_pos', None)
    eef_pos = obs.get('robot0_eef_pos', None)

    if handle_pos is not None and eef_pos is not None:
        dist = np.linalg.norm(np.array(handle_pos) - np.array(eef_pos))
        return dist < HANDLE_REACH_THRESHOLD

    return False


def _check_door_opened(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if door is opened past threshold."""
    # Check hinge joint position
    hinge_qpos = info.get('hinge_qpos', None)
    if hinge_qpos is not None:
        return float(hinge_qpos) > DOOR_OPEN_THRESHOLD

    # Try success flag
    return info.get('success', False)
