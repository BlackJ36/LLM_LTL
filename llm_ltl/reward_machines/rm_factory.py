"""
Factory for creating task-specific Reward Machines.

Provides unified interface to create RMs and get_events functions
for all MAPLE tasks.

Reward Modes (based on IJCAI 2019 "LTL and Beyond" & TRAPs):
- sparse: Only reward at terminal state
- distance: Negative distance to terminal as continuous signal
- progression: Discrete reward for state transitions
- hybrid: Progression + potential-based shaping (recommended)
"""

from typing import Callable, Dict, Set, Any, Tuple, Optional

from llm_ltl.reward_machines.reward_machine import RewardMachine, CompositeRewardMachine

# Import task-specific proposition modules
from llm_ltl.reward_machines.propositions import lift_props
from llm_ltl.reward_machines.propositions import door_props
from llm_ltl.reward_machines.propositions import pnp_props
from llm_ltl.reward_machines.propositions import wipe_props
from llm_ltl.reward_machines.propositions import stack_props
from llm_ltl.reward_machines.propositions import nut_props
from llm_ltl.reward_machines.propositions import cleanup_props
from llm_ltl.reward_machines.propositions import peg_props


# Type alias for event functions
EventFn = Callable[[Dict[str, Any], Dict[str, Any]], Set[str]]

# Default reward shaping configuration
DEFAULT_REWARD_CONFIG = {
    'reward_mode': 'hybrid',
    'gamma': 0.99,
    'use_potential_shaping': True,
    'potential_scale': 0.1,
    'terminal_reward': 1.0,
}


def create_rm(
    task_name: str,
    reward_mode: str = 'hybrid',
    gamma: float = 0.99,
    use_potential_shaping: bool = True,
    potential_scale: float = 0.1,
    terminal_reward: float = 1.0,
    enable_state_validation: bool = True,
    default_regression_penalty: float = 0.1,
) -> RewardMachine:
    """
    Create a Reward Machine for the specified task.

    Args:
        task_name: One of 'lift', 'door', 'pnp', 'wipe', 'stack', 'nut', 'cleanup', 'peg_ins'
        reward_mode: 'sparse', 'distance', 'progression', or 'hybrid' (default)
        gamma: Discount factor for potential-based shaping
        use_potential_shaping: Whether to use potential-based shaping in hybrid mode
        potential_scale: Scale for potential reward component
        terminal_reward: Bonus for reaching terminal state
        enable_state_validation: Whether to enable state validation with regression
        default_regression_penalty: Default penalty for state regression

    Returns:
        Configured RewardMachine instance

    Raises:
        ValueError: If task_name is not recognized

    Example:
        # Default hybrid mode with state validation (recommended)
        rm = create_rm('stack')

        # Sparse mode (only terminal reward)
        rm = create_rm('stack', reward_mode='sparse')

        # Disable state validation
        rm = create_rm('stack', enable_state_validation=False)

        # Custom regression penalty
        rm = create_rm('stack', default_regression_penalty=0.2)
    """
    task_name = task_name.lower()

    # Common kwargs for all RMs
    reward_kwargs = {
        'reward_mode': reward_mode,
        'gamma': gamma,
        'use_potential_shaping': use_potential_shaping,
        'potential_scale': potential_scale,
        'terminal_reward': terminal_reward,
        'enable_state_validation': enable_state_validation,
        'default_regression_penalty': default_regression_penalty,
    }

    if task_name == 'lift':
        return _create_lift_rm(**reward_kwargs)
    elif task_name == 'door':
        return _create_door_rm(**reward_kwargs)
    elif task_name in ('pnp', 'pick_place', 'pickplace'):
        return _create_pnp_rm(**reward_kwargs)
    elif task_name == 'wipe':
        return _create_wipe_rm(**reward_kwargs)
    elif task_name == 'stack':
        return _create_stack_rm(**reward_kwargs)
    elif task_name in ('nut', 'nut_assembly'):
        return _create_nut_rm(**reward_kwargs)
    elif task_name == 'cleanup':
        return _create_cleanup_rm(**reward_kwargs)
    elif task_name in ('peg_ins', 'peg_in_hole', 'peg'):
        return _create_peg_rm(**reward_kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}. "
                        f"Available: lift, door, pnp, wipe, stack, nut, cleanup, peg_ins")


def get_events_fn(task_name: str) -> EventFn:
    """
    Get the event detection function for the specified task.

    Args:
        task_name: Task name

    Returns:
        Function that takes (obs, info) and returns Set[str] of events
    """
    task_name = task_name.lower()

    event_fns = {
        'lift': lift_props.get_events,
        'door': door_props.get_events,
        'pnp': pnp_props.get_events,
        'pick_place': pnp_props.get_events,
        'pickplace': pnp_props.get_events,
        'wipe': wipe_props.get_events,
        'stack': stack_props.get_events,
        'nut': nut_props.get_events,
        'nut_assembly': nut_props.get_events,
        'cleanup': cleanup_props.get_events,
        'peg_ins': peg_props.get_events,
        'peg_in_hole': peg_props.get_events,
        'peg': peg_props.get_events,
    }

    if task_name not in event_fns:
        raise ValueError(f"Unknown task: {task_name}")

    return event_fns[task_name]


def create_rm_and_events(
    task_name: str,
    reward_mode: str = 'hybrid',
    **reward_kwargs
) -> Tuple[RewardMachine, EventFn]:
    """
    Create both RM and events function for a task.

    Args:
        task_name: Task name
        reward_mode: Reward computation mode
        **reward_kwargs: Additional reward configuration

    Returns:
        Tuple of (RewardMachine, events_function)
    """
    return create_rm(task_name, reward_mode=reward_mode, **reward_kwargs), get_events_fn(task_name)


# ==================== Task-specific RM creators ====================

def _create_lift_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Lift task.

    LTL: ◇(grasped ∧ ◇lifted)
    States: u0 (init) -> u1 (grasped) -> u2 (lifted/terminal)

    State validation:
    - u1 requires 'cube_grasped' to be maintained
    - If cube dropped, regress to u0
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2'],
        initial_state='u0',
        transitions={
            ('u0', 'cube_grasped'): 'u1',
            ('u1', 'cube_lifted'): 'u2',
        },
        rewards={
            ('u0', 'u1'): 0.3,  # Grasp reward
            ('u1', 'u2'): 0.7,  # Lift reward
        },
        terminal_states=['u2'],
        name='Lift_RM',
        state_validators={
            # u1: must have cube_grasped, else regress to u0
            'u1': ({'cube_grasped'}, 'u0', penalty),
        },
        **reward_kwargs
    )


def _create_door_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Door task.

    LTL: ◇(handle_reached ∧ ◇door_opened)
    States: u0 -> u1 (reached) -> u2 (opened/terminal)

    State validation:
    - u1 requires 'handle_reached' to be maintained
    - If hand moves away, regress to u0
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2'],
        initial_state='u0',
        transitions={
            ('u0', 'handle_reached'): 'u1',
            ('u1', 'door_opened'): 'u2',
        },
        rewards={
            ('u0', 'u1'): 0.3,  # Reach handle
            ('u1', 'u2'): 0.7,  # Open door
        },
        terminal_states=['u2'],
        name='Door_RM',
        state_validators={
            # u1: must have handle_reached, else regress to u0
            'u1': ({'handle_reached'}, 'u0', penalty),
        },
        **reward_kwargs
    )


def _create_pnp_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Pick and Place task.

    LTL: ◇(grasped ∧ ◇(lifted ∧ ◇(above_bin ∧ ◇in_bin)))
    States: u0 -> u1 -> u2 -> u3 -> u4 (terminal)

    State validation:
    - u1, u2, u3 require 'obj_grasped' to be maintained
    - If object dropped before reaching bin, regress appropriately
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2', 'u3', 'u4'],
        initial_state='u0',
        transitions={
            ('u0', 'obj_grasped'): 'u1',
            ('u1', 'obj_lifted'): 'u2',
            ('u2', 'obj_above_bin'): 'u3',
            ('u3', 'obj_in_bin'): 'u4',
        },
        rewards={
            ('u0', 'u1'): 0.2,  # Grasp
            ('u1', 'u2'): 0.2,  # Lift
            ('u2', 'u3'): 0.2,  # Move above bin
            ('u3', 'u4'): 0.4,  # Place in bin
        },
        terminal_states=['u4'],
        name='PNP_RM',
        state_validators={
            # u1: must have obj_grasped
            'u1': ({'obj_grasped'}, 'u0', penalty),
            # u2: must have obj_grasped and obj_lifted
            'u2': ({'obj_grasped'}, 'u0', penalty * 1.5),
            # u3: must have obj_grasped (still holding above bin)
            'u3': ({'obj_grasped'}, 'u0', penalty * 2.0),
        },
        **reward_kwargs
    )


def _create_wipe_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Wipe task.

    LTL: ◇(contact ∧ ◇(partial ∧ ◇(major ∧ ◇complete)))
    States: u0 -> u1 (contact) -> u2 (25%) -> u3 (75%) -> u4 (100%/terminal)

    State validation:
    - Wipe progress is generally monotonic (can't un-wipe)
    - Only u1 requires contact to be maintained
    - Progress states (u2, u3) are sticky (no regression)
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2', 'u3', 'u4'],
        initial_state='u0',
        transitions={
            ('u0', 'contact_made'): 'u1',
            ('u1', 'partial_wipe'): 'u2',
            ('u2', 'major_wipe'): 'u3',
            ('u3', 'wipe_complete'): 'u4',
        },
        rewards={
            ('u0', 'u1'): 0.2,  # Make contact
            ('u1', 'u2'): 0.2,  # 25% wiped
            ('u2', 'u3'): 0.3,  # 75% wiped
            ('u3', 'u4'): 0.3,  # 100% wiped
        },
        terminal_states=['u4'],
        name='Wipe_RM',
        state_validators={
            # u1: must maintain contact
            'u1': ({'contact_made'}, 'u0', penalty),
            # u2, u3: wipe progress is sticky (no validators needed)
        },
        **reward_kwargs
    )


def _create_stack_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Stack task.

    LTL: ◇(grasped ∧ ◇(lifted ∧ ◇(aligned ∧ ◇stacked)))
    States: u0 -> u1 (grasped) -> u2 (lifted) -> u3 (aligned) -> u4 (stacked/terminal)

    State validation:
    - u1: must have cubeA_grasped, else regress to u0
    - u2: must have cubeA_grasped AND cubeA_lifted, else regress
    - u3: must have cubeA_grasped AND cubes_aligned, else regress
    - If cube dropped at any stage, regress appropriately
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2', 'u3', 'u4'],
        initial_state='u0',
        transitions={
            ('u0', 'cubeA_grasped'): 'u1',
            ('u1', 'cubeA_lifted'): 'u2',
            ('u2', 'cubes_aligned'): 'u3',
            ('u3', 'stacked'): 'u4',
        },
        rewards={
            ('u0', 'u1'): 0.2,  # Grasp red cube
            ('u1', 'u2'): 0.2,  # Lift
            ('u2', 'u3'): 0.3,  # Align above green
            ('u3', 'u4'): 0.3,  # Stack and release
        },
        terminal_states=['u4'],
        name='Stack_RM',
        state_validators={
            # u1: must have cubeA_grasped
            'u1': ({'cubeA_grasped'}, 'u0', penalty),
            # u2: must have cubeA_grasped (lifted implies grasped maintained)
            'u2': ({'cubeA_grasped'}, 'u0', penalty * 1.5),
            # u3: must have cubeA_grasped (still holding while aligned)
            'u3': ({'cubeA_grasped'}, 'u0', penalty * 2.0),
        },
        **reward_kwargs
    )


def _create_nut_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Nut Assembly task.

    LTL: ◇(grasped ∧ ◇(lifted ∧ ◇(above_peg ∧ ◇on_peg)))
    States: u0 -> u1 -> u2 -> u3 -> u4 (terminal)

    State validation:
    - u1, u2, u3 require nut_grasped to be maintained
    - If nut dropped before reaching peg, regress appropriately
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2', 'u3', 'u4'],
        initial_state='u0',
        transitions={
            ('u0', 'nut_grasped'): 'u1',
            ('u1', 'nut_lifted'): 'u2',
            ('u2', 'nut_above_peg'): 'u3',
            ('u3', 'nut_on_peg'): 'u4',
        },
        rewards={
            ('u0', 'u1'): 0.2,  # Grasp nut
            ('u1', 'u2'): 0.3,  # Lift
            ('u2', 'u3'): 0.2,  # Move above peg
            ('u3', 'u4'): 0.3,  # Place on peg
        },
        terminal_states=['u4'],
        name='Nut_RM',
        state_validators={
            # u1: must have nut_grasped
            'u1': ({'nut_grasped'}, 'u0', penalty),
            # u2: must have nut_grasped
            'u2': ({'nut_grasped'}, 'u0', penalty * 1.5),
            # u3: must have nut_grasped (still holding above peg)
            'u3': ({'nut_grasped'}, 'u0', penalty * 2.0),
        },
        **reward_kwargs
    )


def _create_cleanup_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Cleanup task (sequential PNP then Push).

    LTL: ◇(pnp_done ∧ ◇push_done)
    States: u0 -> u1 (grasped) -> u2 (in_bin) -> u3 (push_contact) -> u4 (push_done/terminal)

    State validation:
    - u1: must have pnp_grasped to be maintained
    - u2: pnp object in bin is sticky (no regression)
    - u3: must have push_contact maintained
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2', 'u3', 'u4'],
        initial_state='u0',
        transitions={
            ('u0', 'pnp_grasped'): 'u1',
            ('u1', 'pnp_in_bin'): 'u2',
            ('u2', 'push_contact'): 'u3',
            ('u3', 'push_complete'): 'u4',
        },
        rewards={
            ('u0', 'u1'): 0.15,  # Grasp PNP object
            ('u1', 'u2'): 0.35,  # Place in bin
            ('u2', 'u3'): 0.15,  # Contact push object
            ('u3', 'u4'): 0.35,  # Complete push
        },
        terminal_states=['u4'],
        name='Cleanup_RM',
        state_validators={
            # u1: must have pnp_grasped
            'u1': ({'pnp_grasped'}, 'u0', penalty),
            # u2: pnp in bin is sticky (no validator)
            # u3: must maintain push_contact
            'u3': ({'push_contact'}, 'u2', penalty),
        },
        **reward_kwargs
    )


def _create_peg_rm(**reward_kwargs) -> RewardMachine:
    """
    Create RM for Peg Insertion task.

    LTL: ◇(grasped ∧ ◇(aligned ∧ ◇(positioned ∧ ◇inserted)))
    States: u0 -> u1 -> u2 -> u3 -> u4 (terminal)

    State validation:
    - u1, u2, u3 require peg_grasped to be maintained
    - If peg dropped before insertion, regress appropriately
    """
    penalty = reward_kwargs.get('default_regression_penalty', 0.1)

    return RewardMachine(
        states=['u0', 'u1', 'u2', 'u3', 'u4'],
        initial_state='u0',
        transitions={
            ('u0', 'peg_grasped'): 'u1',
            ('u1', 'peg_aligned'): 'u2',
            ('u2', 'peg_positioned'): 'u3',
            ('u3', 'peg_inserted'): 'u4',
        },
        rewards={
            ('u0', 'u1'): 0.15,  # Grasp peg
            ('u1', 'u2'): 0.35,  # Align orientation
            ('u2', 'u3'): 0.25,  # Position near hole
            ('u3', 'u4'): 0.25,  # Insert
        },
        terminal_states=['u4'],
        name='PegIns_RM',
        state_validators={
            # u1: must have peg_grasped
            'u1': ({'peg_grasped'}, 'u0', penalty),
            # u2: must have peg_grasped
            'u2': ({'peg_grasped'}, 'u0', penalty * 1.5),
            # u3: must have peg_grasped (still holding while positioning)
            'u3': ({'peg_grasped'}, 'u0', penalty * 2.0),
        },
        **reward_kwargs
    )


# ==================== Utility functions ====================

def get_available_tasks() -> list:
    """Get list of available task names."""
    return ['lift', 'door', 'pnp', 'wipe', 'stack', 'nut', 'cleanup', 'peg_ins']


def get_available_reward_modes() -> list:
    """Get list of available reward modes."""
    return ['sparse', 'distance', 'progression', 'hybrid']


def get_task_info(task_name: str) -> dict:
    """Get information about a task's RM structure."""
    rm = create_rm(task_name)
    return {
        'name': rm.name,
        'n_states': rm.n_states,
        'states': sorted(rm.states),
        'transitions': list(rm.transitions.keys()),
        'terminal_states': list(rm.terminal_states),
        'total_transition_reward': sum(rm.rewards.values()),
        'distances': {s: rm.distance_to_terminal(s) for s in sorted(rm.states)},
        'potentials': {s: rm.potential(s) for s in sorted(rm.states)},
    }


def print_rm_summary(task_name: str):
    """Print a summary of the RM for a task."""
    rm = create_rm(task_name)
    info = get_task_info(task_name)

    print(f"\n{'='*50}")
    print(f"Reward Machine: {rm.name}")
    print(f"{'='*50}")
    print(f"States: {info['states']}")
    print(f"Terminal: {info['terminal_states']}")
    print(f"\nTransitions:")
    for (from_s, event), to_s in rm.transitions.items():
        reward = rm.rewards.get((from_s, to_s), 0)
        print(f"  {from_s} --[{event}]--> {to_s}  (reward: {reward})")

    print(f"\nDistance to terminal:")
    for state in sorted(rm.states):
        dist = rm.distance_to_terminal(state)
        pot = rm.potential(state)
        print(f"  {state}: dist={dist}, Φ={pot:.2f}")

    print(f"\nReward mode: {rm.reward_mode}")
    print(f"Potential shaping: {rm.use_potential_shaping}")
    print(f"Potential scale: {rm.potential_scale}")
    print(f"{'='*50}\n")
