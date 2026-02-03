"""
Base proposition functions shared across tasks.

These provide common checks for grasping, lifting, distance, etc.
"""

from typing import Any, Dict, Optional
import numpy as np


def check_grasped(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    gripper_threshold: float = 0.02
) -> bool:
    """
    Check if gripper is closed (holding something).

    Args:
        obs: Observation dict containing 'robot0_gripper_qpos'
        info: Info dict (unused for base check)
        gripper_threshold: Max gripper opening to consider closed

    Returns:
        True if gripper is closed
    """
    gripper_qpos = obs.get('robot0_gripper_qpos', None)
    if gripper_qpos is None:
        return False

    # For parallel-jaw gripper, qpos[0] is the opening distance
    # Smaller value = more closed
    if isinstance(gripper_qpos, np.ndarray):
        opening = gripper_qpos[0]
    else:
        opening = gripper_qpos

    return opening < gripper_threshold


def check_lifted(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    obj_pos_key: str,
    table_height: float = 0.8,
    lift_threshold: float = 0.04
) -> bool:
    """
    Check if object is lifted above table.

    Args:
        obs: Observation dict
        info: Info dict containing object position
        obj_pos_key: Key for object position in info (e.g., 'cube_pos')
        table_height: Height of table surface
        lift_threshold: Minimum height above table to be considered lifted

    Returns:
        True if object is above threshold
    """
    obj_pos = info.get(obj_pos_key, None)
    if obj_pos is None:
        # Try obs
        obj_pos = obs.get(obj_pos_key, None)
    if obj_pos is None:
        return False

    if isinstance(obj_pos, np.ndarray):
        obj_z = obj_pos[2]
    else:
        obj_z = obj_pos

    return obj_z > table_height + lift_threshold


def check_distance(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    pos1_key: str,
    pos2_key: str,
    threshold: float,
    use_2d: bool = False
) -> bool:
    """
    Check if distance between two positions is below threshold.

    Args:
        obs: Observation dict
        info: Info dict
        pos1_key: Key for first position
        pos2_key: Key for second position
        threshold: Distance threshold
        use_2d: If True, only use x-y distance (ignore z)

    Returns:
        True if distance < threshold
    """
    pos1 = _get_position(obs, info, pos1_key)
    pos2 = _get_position(obs, info, pos2_key)

    if pos1 is None or pos2 is None:
        return False

    if use_2d:
        dist = np.linalg.norm(pos1[:2] - pos2[:2])
    else:
        dist = np.linalg.norm(pos1 - pos2)

    return dist < threshold


def check_contact(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    contact_key: str = 'contact'
) -> bool:
    """
    Check if contact is detected.

    Args:
        obs: Observation dict
        info: Info dict containing contact flag
        contact_key: Key for contact flag

    Returns:
        True if contact detected
    """
    return bool(info.get(contact_key, False))


def check_in_bounds(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    obj_pos_key: str,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray
) -> bool:
    """
    Check if object position is within bounds.

    Args:
        obs: Observation dict
        info: Info dict
        obj_pos_key: Key for object position
        bounds_low: Lower bounds [x, y, z]
        bounds_high: Upper bounds [x, y, z]

    Returns:
        True if object is within bounds
    """
    obj_pos = _get_position(obs, info, obj_pos_key)
    if obj_pos is None:
        return False

    return np.all(obj_pos >= bounds_low) and np.all(obj_pos <= bounds_high)


def check_angle_aligned(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    cos_key: str,
    threshold: float = 0.95
) -> bool:
    """
    Check if angle alignment (cosine similarity) is above threshold.

    Args:
        obs: Observation dict
        info: Info dict containing cosine value
        cos_key: Key for cosine value
        threshold: Minimum cosine for alignment (0.95 ≈ 18°)

    Returns:
        True if aligned
    """
    cos_val = info.get(cos_key, None)
    if cos_val is None:
        cos_val = obs.get(cos_key, None)
    if cos_val is None:
        return False

    return float(cos_val) > threshold


def _get_position(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    key: str
) -> Optional[np.ndarray]:
    """Helper to get position from obs or info."""
    pos = info.get(key, None)
    if pos is None:
        pos = obs.get(key, None)
    if pos is None:
        return None

    return np.array(pos) if not isinstance(pos, np.ndarray) else pos


def compute_horizontal_distance(
    obs: Dict[str, Any],
    info: Dict[str, Any],
    pos1_key: str,
    pos2_key: str
) -> float:
    """Compute horizontal (x-y) distance between two positions."""
    pos1 = _get_position(obs, info, pos1_key)
    pos2 = _get_position(obs, info, pos2_key)

    if pos1 is None or pos2 is None:
        return float('inf')

    return float(np.linalg.norm(pos1[:2] - pos2[:2]))
