"""
Lift task propositions.

Task: Lift a cube above the table.
LTL: ◇(cube_grasped ∧ ◇cube_lifted)
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite lift.py
TABLE_HEIGHT = 0.8
LIFT_THRESHOLD = 0.04  # 4cm above table
GRIPPER_CLOSED_THRESHOLD = 0.02


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Lift task.

    Propositions:
    - cube_grasped: Gripper is holding the cube
    - cube_lifted: Cube is above table + threshold

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check cube_grasped
    if _check_cube_grasped(obs, info):
        events.add('cube_grasped')

    # Check cube_lifted
    if _check_cube_lifted(obs, info):
        events.add('cube_lifted')

    return events


def _check_cube_grasped(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if gripper is holding the cube."""
    # Method 1: Check gripper qpos (closed = holding)
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos

        # Gripper is closed
        if gripper_opening < GRIPPER_CLOSED_THRESHOLD:
            # Also check that cube is near gripper
            cube_pos = _get_cube_pos(obs, info)
            eef_pos = obs.get('robot0_eef_pos', None)
            if cube_pos is not None and eef_pos is not None:
                dist = np.linalg.norm(np.array(cube_pos) - np.array(eef_pos))
                if dist < 0.05:  # Within 5cm
                    return True

    # Method 2: Use info flag if available
    return info.get('grasp', False) or info.get('grasped', False)


def _check_cube_lifted(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if cube is lifted above table."""
    cube_pos = _get_cube_pos(obs, info)
    if cube_pos is None:
        return False

    cube_z = cube_pos[2] if isinstance(cube_pos, (list, np.ndarray)) else cube_pos
    return cube_z > TABLE_HEIGHT + LIFT_THRESHOLD


def _get_cube_pos(obs: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
    """Get cube position from obs or info."""
    # Try info first (more reliable)
    for key in ['cube_pos', 'object_pos', 'Cube_pos']:
        if key in info:
            return np.array(info[key])

    # Try object-state in obs
    obj_state = obs.get('object-state', None)
    if obj_state is not None:
        # First 3 values are typically position
        return np.array(obj_state[:3])

    return None
