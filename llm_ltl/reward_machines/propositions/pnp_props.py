"""
Pick and Place task propositions.

Task: Pick up object(s) and place in bin(s).
LTL: ◇(obj_grasped ∧ ◇(obj_lifted ∧ ◇obj_in_bin))
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite pick_place.py
TABLE_HEIGHT = 0.8
LIFT_THRESHOLD = 0.1  # 10cm above table
GRIPPER_CLOSED_THRESHOLD = 0.02
BIN_HOVER_THRESHOLD = 0.1  # 10cm horizontal distance to bin
BIN_PLACE_THRESHOLD = 0.05  # 5cm height above bin base


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Pick and Place task.

    Propositions:
    - obj_grasped: Gripper is holding an object
    - obj_lifted: Object is lifted above table
    - obj_above_bin: Object is horizontally above target bin
    - obj_in_bin: Object is inside bin

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check obj_grasped
    if _check_obj_grasped(obs, info):
        events.add('obj_grasped')

    # Check obj_lifted
    if _check_obj_lifted(obs, info):
        events.add('obj_lifted')

    # Check obj_above_bin
    if _check_obj_above_bin(obs, info):
        events.add('obj_above_bin')

    # Check obj_in_bin
    if _check_obj_in_bin(obs, info):
        events.add('obj_in_bin')

    return events


def _check_obj_grasped(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if any object is grasped."""
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos
        return gripper_opening < GRIPPER_CLOSED_THRESHOLD

    return info.get('grasp', False) or info.get('grasped', False)


def _check_obj_lifted(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if object is lifted above table."""
    # Try to get object positions from info
    for key in ['Can_pos', 'Milk_pos', 'Bread_pos', 'Cereal_pos', 'object_pos']:
        if key in info:
            obj_pos = np.array(info[key])
            if obj_pos[2] > TABLE_HEIGHT + LIFT_THRESHOLD:
                return True

    # Check object-state
    obj_state = obs.get('object-state', None)
    if obj_state is not None:
        # Typically first 3 values are position
        obj_z = obj_state[2]
        return obj_z > TABLE_HEIGHT + LIFT_THRESHOLD

    return False


def _check_obj_above_bin(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if object is horizontally above bin."""
    # This requires knowing which bin to use
    # For now, use a simple heuristic based on position
    obj_state = obs.get('object-state', None)
    if obj_state is None:
        return False

    obj_xy = np.array(obj_state[:2])

    # Bins are typically at specific x positions (check pick_place.py)
    # This is a simplified check
    bin_x_range = (-0.3, 0.3)
    bin_y_range = (0.1, 0.4)

    return (bin_x_range[0] < obj_xy[0] < bin_x_range[1] and
            bin_y_range[0] < obj_xy[1] < bin_y_range[1])


def _check_obj_in_bin(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if object is placed in bin."""
    # Use info flag if available
    if 'objects_in_bins' in info:
        return np.any(info['objects_in_bins'])

    return info.get('success', False)
