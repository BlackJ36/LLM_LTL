"""
Tests for LTL Reward Machine implementation.

Tests cover:
- Basic RM functionality (transitions, reset, one-hot)
- BFS distance computation
- Potential-based reward shaping
- Multiple reward modes (sparse, distance, progression, hybrid)

Run with: PYTHONPATH=. pytest tests/test_reward_machines.py -v
"""

import pytest
import numpy as np

from llm_ltl.reward_machines import (
    RewardMachine,
    create_rm,
    get_events_fn,
    get_available_tasks,
    get_task_info,
)
from llm_ltl.reward_machines.rm_factory import get_available_reward_modes


class TestRewardMachine:
    """Tests for base RewardMachine class."""

    def test_create_simple_rm(self):
        """Test creating a simple RM."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'event_a'): 'u1',
                ('u1', 'event_b'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.5,
                ('u1', 'u2'): 0.5,
            },
            terminal_states=['u2'],
            name='TestRM'
        )

        assert rm.current_state == 'u0'
        assert rm.n_states == 3
        assert not rm.is_terminal()

    def test_rm_transitions(self):
        """Test RM state transitions."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'grasped'): 'u1',
                ('u1', 'lifted'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='progression',  # Use progression for predictable rewards
        )

        # Initial state
        assert rm.current_state == 'u0'

        # No matching event
        state, reward, terminal = rm.step({'random_event'})
        assert state == 'u0'
        assert reward == 0.0
        assert not terminal

        # Transition on 'grasped'
        state, reward, terminal = rm.step({'grasped'})
        assert state == 'u1'
        assert reward == 0.3
        assert not terminal

        # Transition on 'lifted' (includes terminal bonus)
        state, reward, terminal = rm.step({'lifted'})
        assert state == 'u2'
        assert reward == 0.7 + 1.0  # transition + terminal_reward
        assert terminal

    def test_rm_reset(self):
        """Test RM reset."""
        rm = RewardMachine(
            states=['u0', 'u1'],
            initial_state='u0',
            transitions={('u0', 'event'): 'u1'},
            rewards={('u0', 'u1'): 1.0},
            terminal_states=['u1'],
        )

        # Move to u1
        rm.step({'event'})
        assert rm.current_state == 'u1'

        # Reset
        state = rm.reset()
        assert state == 'u0'
        assert rm.current_state == 'u0'

    def test_rm_one_hot(self):
        """Test RM state one-hot encoding."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={},
            rewards={},
        )

        one_hot = rm.get_state_one_hot()
        assert one_hot.shape == (3,)
        assert one_hot[0] == 1.0  # u0 is first in sorted order
        assert one_hot.sum() == 1.0


class TestDistanceComputation:
    """Tests for BFS-based distance computation."""

    def test_distance_to_terminal(self):
        """Test distance computation from each state to terminal."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2', 'u3'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
                ('u2', 'c'): 'u3',
            },
            rewards={},
            terminal_states=['u3'],
        )

        # Distances should be computed via BFS
        assert rm.distance_to_terminal('u3') == 0  # Terminal
        assert rm.distance_to_terminal('u2') == 1
        assert rm.distance_to_terminal('u1') == 2
        assert rm.distance_to_terminal('u0') == 3

    def test_potential_function(self):
        """Test potential function Φ(q) = -dist/max_dist."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
            },
            rewards={},
            terminal_states=['u2'],
        )

        # Potentials should be normalized to [-1, 0]
        assert rm.potential('u2') == 0.0  # Terminal has highest potential
        assert rm.potential('u1') == -0.5  # dist=1, max=2
        assert rm.potential('u0') == -1.0  # dist=2, max=2

    def test_progress_based_on_distance(self):
        """Test progress = 1 - (dist / max_dist)."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2', 'u3'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
                ('u2', 'c'): 'u3',
            },
            rewards={},
            terminal_states=['u3'],
        )

        # Progress at u0 (dist=3, max=3)
        assert rm.get_progress() == 0.0

        rm.step({'a'})  # u0 -> u1
        assert abs(rm.get_progress() - 1/3) < 0.01

        rm.step({'b'})  # u1 -> u2
        assert abs(rm.get_progress() - 2/3) < 0.01

        rm.step({'c'})  # u2 -> u3
        assert rm.get_progress() == 1.0


class TestRewardModes:
    """Tests for different reward computation modes."""

    def test_sparse_mode(self):
        """Test sparse reward mode (only terminal reward)."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.5,
                ('u1', 'u2'): 0.5,
            },
            terminal_states=['u2'],
            reward_mode='sparse',
            terminal_reward=1.0,
        )

        # Transition u0 -> u1: no reward in sparse mode
        state, reward, _ = rm.step({'a'})
        assert reward == 0.0

        # Transition u1 -> u2: terminal reward only
        state, reward, _ = rm.step({'b'})
        assert reward == 1.0

    def test_distance_mode(self):
        """Test distance-based reward mode."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
            },
            rewards={},
            terminal_states=['u2'],
            reward_mode='distance',
        )

        # At u0: dist=2, reward = -2/2 = -1.0
        state, reward, _ = rm.step(set())  # No transition
        assert reward == -1.0

        # Transition to u1: dist=1, reward = -1/2 = -0.5
        state, reward, _ = rm.step({'a'})
        assert reward == -0.5

        # Transition to u2: dist=0, reward = 0
        state, reward, _ = rm.step({'b'})
        assert reward == 0.0

    def test_progression_mode(self):
        """Test progression reward mode (transition rewards only)."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='progression',
            terminal_reward=1.0,
        )

        # Transition u0 -> u1: transition reward
        state, reward, _ = rm.step({'a'})
        assert reward == 0.3

        # Transition u1 -> u2: transition + terminal
        state, reward, _ = rm.step({'b'})
        assert reward == 0.7 + 1.0

    def test_hybrid_mode(self):
        """Test hybrid mode (progression + potential shaping)."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='hybrid',
            gamma=0.99,
            potential_scale=0.1,
            terminal_reward=1.0,
        )

        # Transition u0 -> u1
        # Shaping = γ * Φ(u1) - Φ(u0) = 0.99 * (-0.5) - (-1.0) = 0.505
        state, reward, _ = rm.step({'a'})
        expected_shaping = 0.99 * (-0.5) - (-1.0)
        expected_reward = 0.3 + 0.1 * expected_shaping
        assert abs(reward - expected_reward) < 0.01

        # Transition u1 -> u2
        # Shaping = γ * Φ(u2) - Φ(u1) = 0.99 * 0 - (-0.5) = 0.5
        state, reward, _ = rm.step({'b'})
        expected_shaping = 0.99 * 0 - (-0.5)
        expected_reward = 0.7 + 0.1 * expected_shaping + 1.0  # + terminal
        assert abs(reward - expected_reward) < 0.01


class TestRMFactory:
    """Tests for RM factory functions."""

    def test_available_tasks(self):
        """Test getting available tasks."""
        tasks = get_available_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 8
        assert 'lift' in tasks
        assert 'stack' in tasks
        assert 'door' in tasks

    def test_available_reward_modes(self):
        """Test getting available reward modes."""
        modes = get_available_reward_modes()
        assert 'sparse' in modes
        assert 'distance' in modes
        assert 'progression' in modes
        assert 'hybrid' in modes

    def test_create_all_tasks(self):
        """Test creating RMs for all tasks."""
        for task in get_available_tasks():
            rm = create_rm(task)
            assert isinstance(rm, RewardMachine)
            assert rm.n_states >= 3  # All tasks have at least 3 states
            assert len(rm.terminal_states) >= 1

    def test_create_with_different_modes(self):
        """Test creating RMs with different reward modes."""
        for mode in get_available_reward_modes():
            rm = create_rm('stack', reward_mode=mode)
            assert rm.reward_mode == mode

    def test_create_lift_rm(self):
        """Test Lift task RM."""
        rm = create_rm('lift', reward_mode='progression')
        assert rm.name == 'Lift_RM'
        assert rm.n_states == 3
        assert 'u2' in rm.terminal_states

        # Check distances
        assert rm.distance_to_terminal('u0') == 2
        assert rm.distance_to_terminal('u1') == 1
        assert rm.distance_to_terminal('u2') == 0

    def test_create_stack_rm(self):
        """Test Stack task RM."""
        rm = create_rm('stack')
        assert rm.name == 'Stack_RM'
        assert rm.n_states == 5

        # Check distances
        assert rm.distance_to_terminal('u0') == 4
        assert rm.distance_to_terminal('u4') == 0

    def test_get_events_fn(self):
        """Test getting events functions."""
        for task in get_available_tasks():
            events_fn = get_events_fn(task)
            assert callable(events_fn)

    def test_task_info(self):
        """Test getting task info."""
        for task in get_available_tasks():
            info = get_task_info(task)
            assert 'name' in info
            assert 'n_states' in info
            assert 'distances' in info
            assert 'potentials' in info


class TestPropositions:
    """Tests for task-specific proposition functions."""

    def test_lift_propositions(self):
        """Test Lift task propositions."""
        from llm_ltl.reward_machines.propositions import lift_props

        # Test with empty obs/info
        events = lift_props.get_events({}, {})
        assert isinstance(events, set)

        # Test with grasped cube
        obs = {
            'robot0_gripper_qpos': np.array([0.01, 0.01]),  # Closed
            'robot0_eef_pos': np.array([0.0, 0.0, 0.85]),
        }
        info = {
            'cube_pos': np.array([0.0, 0.0, 0.85]),  # Near gripper
        }
        events = lift_props.get_events(obs, info)
        assert 'cube_grasped' in events

        # Test with lifted cube
        info['cube_pos'] = np.array([0.0, 0.0, 0.9])  # Above table (0.8 + 0.04)
        events = lift_props.get_events(obs, info)
        assert 'cube_lifted' in events

    def test_stack_propositions(self):
        """Test Stack task propositions."""
        from llm_ltl.reward_machines.propositions import stack_props

        # Test stacked configuration
        obs = {
            'robot0_gripper_qpos': np.array([0.04, 0.04]),  # Open (released)
            'object-state': np.concatenate([
                [0.0, 0.0, 0.88, 1, 0, 0, 0],  # cubeA at height 0.88
                [0.0, 0.0, 0.82, 1, 0, 0, 0],  # cubeB at height 0.82
            ]),
        }
        info = {
            'cubeA_pos': np.array([0.0, 0.0, 0.88]),
            'cubeB_pos': np.array([0.0, 0.0, 0.82]),
        }
        events = stack_props.get_events(obs, info)

        # Check that cubes are aligned (same x, y)
        assert 'cubes_aligned' in events

    def test_door_propositions(self):
        """Test Door task propositions."""
        from llm_ltl.reward_machines.propositions import door_props

        # Test door opened
        info = {
            'hinge_qpos': 0.4,  # > 0.3 threshold
        }
        events = door_props.get_events({}, info)
        assert 'door_opened' in events


class TestRMIntegration:
    """Integration tests for RM with events functions."""

    def test_lift_integration_hybrid(self):
        """Test Lift RM with hybrid reward mode."""
        rm = create_rm('lift', reward_mode='hybrid', potential_scale=0.1)
        events_fn = get_events_fn('lift')

        # Simulate episode
        obs = {'robot0_gripper_qpos': np.array([0.04, 0.04])}
        info = {'cube_pos': np.array([0.0, 0.0, 0.82])}

        # Initial progress
        assert rm.get_progress() == 0.0
        assert rm.distance_to_terminal() == 2

        # Grasp cube
        obs['robot0_gripper_qpos'] = np.array([0.01, 0.01])
        obs['robot0_eef_pos'] = np.array([0.0, 0.0, 0.82])
        events = events_fn(obs, info)
        state, reward, _ = rm.step(events)

        assert state == 'u1'
        assert rm.distance_to_terminal() == 1
        assert rm.get_progress() == 0.5
        assert reward > 0  # Transition + positive shaping

        # Lift cube (update BOTH cube_pos AND eef_pos to maintain grasp)
        info['cube_pos'] = np.array([0.0, 0.0, 0.90])
        obs['robot0_eef_pos'] = np.array([0.0, 0.0, 0.90])  # EEF moves with cube
        events = events_fn(obs, info)
        state, reward, terminal = rm.step(events)

        assert state == 'u2'
        assert terminal
        assert rm.get_progress() == 1.0
        assert reward > 1.0  # Transition + shaping + terminal


class TestStateValidation:
    """Tests for state validation and regression mechanism."""

    def test_state_validation_basic(self):
        """Test basic state validation with regression."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'grasped'): 'u1',
                ('u1', 'lifted'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='progression',
            state_validators={
                # u1 requires 'grasped' to be maintained
                'u1': ({'grasped'}, 'u0', 0.1),
            },
            enable_state_validation=True,
        )

        # Transition to u1
        state, reward, _ = rm.step({'grasped'})
        assert state == 'u1'
        assert reward == 0.3

        # Stay in u1 with grasped maintained
        state, reward, _ = rm.step({'grasped'})
        assert state == 'u1'

        # Now remove 'grasped' - should regress to u0
        state, reward, _ = rm.step(set())  # No events
        assert state == 'u0'
        assert rm._total_regressions == 1

    def test_regression_penalty(self):
        """Test that regression applies penalty."""
        penalty = 0.2
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'grasped'): 'u1',
                ('u1', 'lifted'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='progression',
            state_validators={
                'u1': ({'grasped'}, 'u0', penalty),
            },
            enable_state_validation=True,
        )

        # Move to u1
        rm.step({'grasped'})
        assert rm.current_state == 'u1'

        # Drop - should regress with penalty
        state, reward, _ = rm.step(set())
        assert state == 'u0'
        assert reward == -penalty  # Only regression penalty
        assert rm._total_regression_penalty == penalty

    def test_regression_with_potential_shaping(self):
        """Test regression with hybrid mode (potential shaping)."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'grasped'): 'u1',
                ('u1', 'lifted'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='hybrid',
            gamma=0.99,
            potential_scale=0.1,
            state_validators={
                'u1': ({'grasped'}, 'u0', 0.1),
            },
            enable_state_validation=True,
        )

        # Move to u1
        rm.step({'grasped'})
        assert rm.current_state == 'u1'

        # Drop - potential shaping should also be negative
        # Before: u1, Φ = -0.5
        # After:  u0, Φ = -1.0
        # Shaping = γ * (-1.0) - (-0.5) = -0.99 + 0.5 = -0.49
        state, reward, _ = rm.step(set())
        assert state == 'u0'
        # Reward should be negative (potential shaping + regression penalty)
        assert reward < 0

    def test_no_regression_when_validation_disabled(self):
        """Test that validation can be disabled."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'grasped'): 'u1',
                ('u1', 'lifted'): 'u2',
            },
            rewards={
                ('u0', 'u1'): 0.3,
                ('u1', 'u2'): 0.7,
            },
            terminal_states=['u2'],
            reward_mode='progression',
            state_validators={
                'u1': ({'grasped'}, 'u0', 0.1),
            },
            enable_state_validation=False,  # Disabled!
        )

        # Move to u1
        rm.step({'grasped'})
        assert rm.current_state == 'u1'

        # Remove 'grasped' - should NOT regress because validation is disabled
        state, reward, _ = rm.step(set())
        assert state == 'u1'  # Still in u1
        assert rm._total_regressions == 0

    def test_factory_creates_validators(self):
        """Test that factory creates RMs with state validators."""
        for task in get_available_tasks():
            rm = create_rm(task, enable_state_validation=True)
            # All tasks except wipe should have at least one validator
            if task != 'wipe':
                assert len(rm.state_validators) > 0, f"{task} should have validators"

    def test_stack_regression_scenario(self):
        """Test Stack task regression when cube is dropped."""
        rm = create_rm('stack', reward_mode='hybrid', enable_state_validation=True)

        # Simulate: grasp -> lift -> drop
        # Grasp
        state, _, _ = rm.step({'cubeA_grasped'})
        assert state == 'u1'

        # Lift
        state, _, _ = rm.step({'cubeA_grasped', 'cubeA_lifted'})
        assert state == 'u2'

        # Drop! (no grasped event)
        state, reward, _ = rm.step(set())
        assert state == 'u0'  # Regressed to initial
        assert reward < 0  # Penalty + negative shaping
        assert rm._total_regressions == 1

    def test_multiple_regressions(self):
        """Test multiple regression tracking."""
        rm = RewardMachine(
            states=['u0', 'u1', 'u2'],
            initial_state='u0',
            transitions={
                ('u0', 'a'): 'u1',
                ('u1', 'b'): 'u2',
            },
            rewards={('u0', 'u1'): 0.5, ('u1', 'u2'): 0.5},
            terminal_states=['u2'],
            reward_mode='progression',
            state_validators={
                'u1': ({'a'}, 'u0', 0.1),
            },
        )

        # Cycle: grasp -> drop -> grasp -> drop
        rm.step({'a'})  # u0 -> u1
        rm.step(set())  # u1 -> u0 (regression 1)
        rm.step({'a'})  # u0 -> u1
        rm.step(set())  # u1 -> u0 (regression 2)

        assert rm._total_regressions == 2
        assert rm._regression_counts[('u1', 'u0')] == 2

    def test_diagnostics_include_regression_stats(self):
        """Test that diagnostics include regression statistics."""
        rm = create_rm('lift', enable_state_validation=True)

        # Cause a regression
        rm.step({'cube_grasped'})
        rm.step(set())  # Drop

        diag = rm.get_diagnostics()
        assert 'rm/total_regressions' in diag
        assert 'rm/state_validation_enabled' in diag
        assert diag['rm/total_regressions'] == 1


class TestRegressionIntegration:
    """Integration tests for regression with propositions."""

    def test_lift_drop_recovery(self):
        """Test Lift task: grasp -> lift -> drop -> re-grasp."""
        rm = create_rm('lift', reward_mode='hybrid', enable_state_validation=True)
        events_fn = get_events_fn('lift')

        # Initial
        assert rm.current_state == 'u0'

        # Grasp cube
        obs = {
            'robot0_gripper_qpos': np.array([0.01, 0.01]),  # Closed
            'robot0_eef_pos': np.array([0.0, 0.0, 0.82]),
        }
        info = {'cube_pos': np.array([0.0, 0.0, 0.82])}
        events = events_fn(obs, info)
        state, r1, _ = rm.step(events)
        assert state == 'u1'
        assert r1 > 0  # Positive reward for progress

        # Drop cube (gripper opens, cube falls)
        obs['robot0_gripper_qpos'] = np.array([0.04, 0.04])  # Open
        info['cube_pos'] = np.array([0.0, 0.0, 0.80])  # Fell
        events = events_fn(obs, info)
        state, r2, _ = rm.step(events)
        assert state == 'u0'  # Regressed
        assert r2 < 0  # Negative reward for regression

        # Re-grasp
        obs['robot0_gripper_qpos'] = np.array([0.01, 0.01])
        obs['robot0_eef_pos'] = np.array([0.0, 0.0, 0.80])
        info['cube_pos'] = np.array([0.0, 0.0, 0.80])
        events = events_fn(obs, info)
        state, r3, _ = rm.step(events)
        assert state == 'u1'  # Back to grasped
        assert r3 > 0  # Positive reward again

        # Check total regressions
        assert rm._total_regressions == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
