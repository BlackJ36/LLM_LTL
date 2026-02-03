"""
Nut Assembly task propositions.

Task: Place nut(s) on corresponding peg(s).
LTL: ◇(nut_grasped ∧ ◇(nut_lifted ∧ ◇(nut_above_peg ∧ ◇nut_on_peg)))
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite nut_assembly.py
TABLE_HEIGHT = 0.8
LIFT_THRESHOLD = 0.15  # 15cm above table
GRIPPER_CLOSED_THRESHOLD = 0.02
ABOVE_PEG_THRESHOLD = 0.05  # 5cm horizontal distance to peg
ON_PEG_THRESHOLD = 0.03  # 3cm for x/y, 5cm height


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Nut Assembly task.

    Propositions:
    - nut_grasped: Gripper is holding a nut
    - nut_lifted: Nut is lifted above table
    - nut_above_peg: Nut is horizontally above target peg
    - nut_on_peg: Nut is placed around peg

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check nut_grasped
    if _check_nut_grasped(obs, info):
        events.add('nut_grasped')

    # Check nut_lifted
    if _check_nut_lifted(obs, info):
        events.add('nut_lifted')

    # Check nut_above_peg
    if _check_nut_above_peg(obs, info):
        events.add('nut_above_peg')

    # Check nut_on_peg
    if _check_nut_on_peg(obs, info):
        events.add('nut_on_peg')

    return events


def _check_nut_grasped(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if gripper is holding a nut."""
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos
        return gripper_opening < GRIPPER_CLOSED_THRESHOLD

    return info.get('grasp', False)


def _check_nut_lifted(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if nut is lifted above table."""
    # Try different nut position keys
    for key in ['SquareNut_pos', 'RoundNut_pos', 'nut_pos']:
        if key in info:
            nut_pos = np.array(info[key])
            if nut_pos[2] > TABLE_HEIGHT + LIFT_THRESHOLD:
                return True

    # Try object-state
    obj_state = obs.get('object-state', None)
    if obj_state is not None and len(obj_state) >= 3:
        nut_z = obj_state[2]
        return nut_z > TABLE_HEIGHT + LIFT_THRESHOLD

    return False


def _check_nut_above_peg(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if nut is horizontally above its target peg."""
    # Get nut and peg positions
    nut_pos = None
    peg_pos = None

    for nut_key, peg_key in [('SquareNut_pos', 'peg1_pos'),
                              ('RoundNut_pos', 'peg2_pos')]:
        if nut_key in info and peg_key in info:
            nut_pos = np.array(info[nut_key])
            peg_pos = np.array(info[peg_key])
            break

    if nut_pos is None or peg_pos is None:
        return False

    horizontal_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
    return horizontal_dist < ABOVE_PEG_THRESHOLD


def _check_nut_on_peg(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if nut is placed on peg."""
    # Use info flags
    if 'objects_on_pegs' in info:
        return np.any(info['objects_on_pegs'])

    if info.get('success', False):
        return True

    # Check gripper is open (released)
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos

        # Must be released
        if gripper_opening < GRIPPER_CLOSED_THRESHOLD:
            return False

    return False
