"""
Wipe task propositions.

Task: Wipe markers on a surface using a wiping tool.
LTL: ◇(contact_made ∧ ◇(partial_wipe ∧ ◇(major_wipe ∧ ◇wipe_complete)))
"""

from typing import Any, Dict, Set
import numpy as np

# Default thresholds from robosuite wipe.py
PARTIAL_WIPE_THRESHOLD = 0.25  # 25% wiped
MAJOR_WIPE_THRESHOLD = 0.75   # 75% wiped
COMPLETE_WIPE_THRESHOLD = 1.0  # 100% wiped


def get_events(obs: Dict[str, Any], info: Dict[str, Any]) -> Set[str]:
    """
    Get current true propositions for Wipe task.

    Propositions:
    - contact_made: Wiping tool is in contact with surface
    - partial_wipe: >25% of markers wiped
    - major_wipe: >75% of markers wiped
    - wipe_complete: 100% of markers wiped

    Args:
        obs: Environment observation
        info: Step info dict

    Returns:
        Set of true proposition names
    """
    events = set()

    # Check contact_made
    if _check_contact(obs, info):
        events.add('contact_made')

    # Get wipe progress
    proportion = _get_wipe_proportion(obs, info)

    # Check wipe progress levels
    if proportion >= PARTIAL_WIPE_THRESHOLD:
        events.add('partial_wipe')

    if proportion >= MAJOR_WIPE_THRESHOLD:
        events.add('major_wipe')

    if proportion >= COMPLETE_WIPE_THRESHOLD:
        events.add('wipe_complete')

    return events


def _check_contact(obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
    """Check if wiping tool is in contact with surface."""
    # Check contact flag in info
    if 'contact' in info:
        return bool(info['contact'])

    # Check if any markers have been wiped (implies contact was made)
    proportion = _get_wipe_proportion(obs, info)
    return proportion > 0


def _get_wipe_proportion(obs: Dict[str, Any], info: Dict[str, Any]) -> float:
    """Get proportion of markers wiped."""
    # Try info first
    if 'proportion_wiped' in info:
        return float(info['proportion_wiped'])

    if 'wiped_markers' in info and 'num_markers' in info:
        return len(info['wiped_markers']) / info['num_markers']

    # Check success flag
    if info.get('success', False):
        return 1.0

    return 0.0
