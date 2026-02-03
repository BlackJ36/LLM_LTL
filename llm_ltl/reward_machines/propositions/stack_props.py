"""
Stack task propositions.

Task: Stack red cube (A) on top of green cube (B).
LTL: ◇(cubeA_grasped ∧ ◇(cubeA_lifted ∧ ◇(cubes_aligned ∧ ◇stacked)))
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite stack.py
TABLE_HEIGHT = 0.8
LIFT_THRESHOLD = 0.04  # 4cm above table
ALIGN_THRESHOLD = 0.02  # 2cm horizontal distance for alignment
GRIPPER_CLOSED_THRESHOLD = 0.02
STACK_CONTACT_DIST = 0.04  # Distance for cubes to be "touching"


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Stack task.

    Propositions:
    - cubeA_grasped: Gripper is holding red cube (A)
    - cubeA_lifted: Red cube is above table
    - cubes_aligned: Red cube is horizontally above green cube
    - stacked: Cubes are stacked and red cube released

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    cubeA_pos, cubeB_pos = _get_cube_positions(obs, info)

    # Check cubeA_grasped
    if _check_cubeA_grasped(obs, info, cubeA_pos):
        events.add('cubeA_grasped')

    # Check cubeA_lifted
    if _check_cubeA_lifted(cubeA_pos):
        events.add('cubeA_lifted')

    # Check cubes_aligned
    if _check_cubes_aligned(cubeA_pos, cubeB_pos):
        events.add('cubes_aligned')

    # Check stacked
    if _check_stacked(obs, info, cubeA_pos, cubeB_pos):
        events.add('stacked')

    return events


def _get_cube_positions(obs: Dict[str, Any], info: Dict[str, Any]):
    """Get positions of cubeA (red) and cubeB (green)."""
    cubeA_pos = None
    cubeB_pos = None

    # Try info first
    if 'cubeA_pos' in info:
        cubeA_pos = np.array(info['cubeA_pos'])
    if 'cubeB_pos' in info:
        cubeB_pos = np.array(info['cubeB_pos'])

    # Try object-state from obs
    if cubeA_pos is None or cubeB_pos is None:
        obj_state = obs.get('object-state', None)
        if obj_state is not None:
            # Stack task: first 7 values = cubeA (pos + quat), next 7 = cubeB
            if len(obj_state) >= 14:
                cubeA_pos = np.array(obj_state[:3])
                cubeB_pos = np.array(obj_state[7:10])
            elif len(obj_state) >= 6:
                # Fallback: first 3 = cubeA, next 3 = cubeB
                cubeA_pos = np.array(obj_state[:3])
                cubeB_pos = np.array(obj_state[3:6])

    return cubeA_pos, cubeB_pos


def _check_cubeA_grasped(obs: Dict[str, Any], info: Dict[str, Any],
                          cubeA_pos: np.ndarray) -> bool:
    """Check if gripper is holding cubeA."""
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is None:
        return info.get('grasp', False)

    if isinstance(gripper_qpos, np.ndarray):
        gripper_opening = gripper_qpos[0]
    else:
        gripper_opening = gripper_qpos

    if gripper_opening >= GRIPPER_CLOSED_THRESHOLD:
        return False

    # Check that cubeA is near gripper
    if cubeA_pos is None:
        return False

    eef_pos = obs.get('robot0_eef_pos', None)
    if eef_pos is None:
        return gripper_opening < GRIPPER_CLOSED_THRESHOLD

    dist = np.linalg.norm(cubeA_pos - np.array(eef_pos))
    return dist < 0.05


def _check_cubeA_lifted(cubeA_pos: np.ndarray) -> bool:
    """Check if cubeA is lifted above table."""
    if cubeA_pos is None:
        return False
    return cubeA_pos[2] > TABLE_HEIGHT + LIFT_THRESHOLD


def _check_cubes_aligned(cubeA_pos: np.ndarray, cubeB_pos: np.ndarray) -> bool:
    """Check if cubeA is horizontally aligned above cubeB."""
    if cubeA_pos is None or cubeB_pos is None:
        return False

    # Horizontal distance (x-y only)
    horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
    return horizontal_dist < ALIGN_THRESHOLD


def _check_stacked(obs: Dict[str, Any], info: Dict[str, Any],
                   cubeA_pos: np.ndarray, cubeB_pos: np.ndarray) -> bool:
    """Check if cubes are stacked and cubeA is released."""
    # Check success flag first
    if info.get('success', False):
        return True

    if cubeA_pos is None or cubeB_pos is None:
        return False

    # CubeA should be above cubeB
    if cubeA_pos[2] <= cubeB_pos[2]:
        return False

    # Cubes should be aligned
    horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
    if horizontal_dist > ALIGN_THRESHOLD:
        return False

    # Cubes should be close vertically (touching)
    vertical_dist = cubeA_pos[2] - cubeB_pos[2]
    cube_size = 0.02  # Approximate cube half-size
    if vertical_dist > 2 * cube_size + STACK_CONTACT_DIST:
        return False

    # Gripper should be open (released)
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is not None:
        if isinstance(gripper_qpos, np.ndarray):
            gripper_opening = gripper_qpos[0]
        else:
            gripper_opening = gripper_qpos
        return gripper_opening > GRIPPER_CLOSED_THRESHOLD

    return True
