"""
Reward Machine implementation for LTL-based task specification.

Enhanced with potential-based reward shaping from:
- Icarte et al. "Reward Machines: Exploiting Reward Function Structure in RL" (JAIR 2022)
- Camacho et al. "LTL and Beyond: Formal Languages for Reward Function Specification" (IJCAI 2019)
- TRAPs: "Task-Driven RL with Action Primitives" (IEEE Cybernetics 2024)

Reward Modes:
- sparse: Only reward at terminal state
- distance: Negative distance to terminal as potential
- progression: Reward for state transitions
- hybrid: Combination of progression + potential shaping

State Validation:
- Each non-initial state can have validity conditions
- If conditions are not met, state regresses to fallback state
- Potential-based shaping automatically penalizes regression
"""

from typing import Dict, Set, Tuple, List, Optional, Any
from collections import OrderedDict, deque
import numpy as np


# Type alias for state validators
# Maps state -> (required_events, fallback_state, regression_penalty)
StateValidator = Dict[str, Tuple[Set[str], str, float]]


class RewardMachine:
    """
    Finite State Machine with potential-based reward shaping.

    Supports multiple reward computation modes:
    1. Sparse: R = 1 if terminal else 0
    2. Distance: R = -dist(q, F) (negative distance to accepting states)
    3. Progression: R = transition_reward when state changes
    4. Hybrid: R = transition_reward + γ·Φ(q') - Φ(q)  (progression + potential shaping)

    The potential function Φ(q) = -dist(q, F) encourages moving toward terminal states.
    """

    def __init__(
        self,
        states: List[str],
        initial_state: str,
        transitions: Dict[Tuple[str, str], str],
        rewards: Dict[Tuple[str, str], float],
        terminal_states: Optional[List[str]] = None,
        name: str = "RM",
        # Reward shaping parameters
        reward_mode: str = "hybrid",  # 'sparse', 'distance', 'progression', 'hybrid'
        gamma: float = 0.99,          # Discount for potential shaping
        use_potential_shaping: bool = True,
        potential_scale: float = 0.1,  # Scale for potential-based reward
        terminal_reward: float = 1.0,  # Bonus for reaching terminal
        # State validation parameters
        state_validators: Optional[StateValidator] = None,
        enable_state_validation: bool = True,
        default_regression_penalty: float = 0.1,
    ):
        """
        Initialize the Reward Machine.

        Args:
            states: List of state names (e.g., ['u0', 'u1', 'u2'])
            initial_state: Starting state name
            transitions: Dict mapping (state, event) -> next_state
            rewards: Dict mapping (from_state, to_state) -> reward
            terminal_states: List of terminal/goal states
            name: Name for logging/debugging
            reward_mode: 'sparse', 'distance', 'progression', or 'hybrid'
            gamma: Discount factor for potential-based shaping
            use_potential_shaping: Whether to add potential-based shaping
            potential_scale: Scale factor for potential reward component
            terminal_reward: Bonus reward for reaching terminal state
            state_validators: Dict mapping state -> (required_events, fallback_state, penalty)
                              If required_events not in current events, regress to fallback_state
            enable_state_validation: Whether to enable state validation (default True)
            default_regression_penalty: Default penalty for regression if not specified
        """
        self.states = set(states)
        self.initial_state = initial_state
        self.transitions = transitions
        self.rewards = rewards
        self.terminal_states = set(terminal_states) if terminal_states else set()
        self.name = name

        # Reward shaping parameters
        self.reward_mode = reward_mode
        self.gamma = gamma
        self.use_potential_shaping = use_potential_shaping
        self.potential_scale = potential_scale
        self.terminal_reward = terminal_reward

        # State validation parameters
        self.state_validators = state_validators or {}
        self.enable_state_validation = enable_state_validation
        self.default_regression_penalty = default_regression_penalty

        # Current state tracking
        self.current_state = initial_state

        # Compute distances to terminal states using BFS
        self._distances: Dict[str, int] = {}
        self._compute_distances()

        # Compute potential function Φ(q) = -distance(q, F)
        self._potentials: Dict[str, float] = {}
        self._compute_potentials()

        # Statistics
        self._transition_counts: Dict[Tuple[str, str], int] = {}
        self._state_visits: Dict[str, int] = {s: 0 for s in states}
        self._total_rm_reward = 0.0
        self._total_shaping_reward = 0.0
        self._episode_count = 0

        # Regression statistics
        self._regression_counts: Dict[Tuple[str, str], int] = {}
        self._total_regressions = 0
        self._total_regression_penalty = 0.0

        # Validate
        self._validate()

    def _validate(self):
        """Validate RM configuration."""
        assert self.initial_state in self.states, \
            f"Initial state {self.initial_state} not in states"

        for (state, event), next_state in self.transitions.items():
            assert state in self.states, f"Transition from unknown state: {state}"
            assert next_state in self.states, f"Transition to unknown state: {next_state}"

        for (from_state, to_state), reward in self.rewards.items():
            assert from_state in self.states, f"Reward from unknown state: {from_state}"
            assert to_state in self.states, f"Reward to unknown state: {to_state}"

        assert self.reward_mode in ('sparse', 'distance', 'progression', 'hybrid'), \
            f"Unknown reward mode: {self.reward_mode}"

    def _compute_distances(self):
        """
        Compute minimum distance from each state to any terminal state using BFS.

        Based on: Camacho et al. IJCAI 2019 - "LTL and Beyond"
        """
        # Build reverse adjacency list
        reverse_adj: Dict[str, List[str]] = {s: [] for s in self.states}
        for (from_state, event), to_state in self.transitions.items():
            reverse_adj[to_state].append(from_state)

        # BFS from terminal states
        queue = deque()
        for terminal in self.terminal_states:
            self._distances[terminal] = 0
            queue.append(terminal)

        while queue:
            current = queue.popleft()
            current_dist = self._distances[current]

            for predecessor in reverse_adj[current]:
                if predecessor not in self._distances:
                    self._distances[predecessor] = current_dist + 1
                    queue.append(predecessor)

        # States unreachable from terminal get max distance
        max_dist = len(self.states)
        for state in self.states:
            if state not in self._distances:
                self._distances[state] = max_dist

    def _compute_potentials(self):
        """
        Compute potential function Φ(q) = -distance(q, F).

        Negative distance encourages moving toward terminal states.
        Normalized by max distance for stability.
        """
        if not self._distances:
            for state in self.states:
                self._potentials[state] = 0.0
            return

        max_dist = max(self._distances.values())
        for state, dist in self._distances.items():
            # Φ(q) = -dist / max_dist, so Φ ∈ [-1, 0]
            # Terminal states have Φ = 0, initial has Φ ≈ -1
            if max_dist > 0:
                self._potentials[state] = -dist / max_dist
            else:
                self._potentials[state] = 0.0

    def distance_to_terminal(self, state: Optional[str] = None) -> int:
        """Get minimum distance from state to any terminal state."""
        if state is None:
            state = self.current_state
        return self._distances.get(state, len(self.states))

    def potential(self, state: Optional[str] = None) -> float:
        """Get potential function value Φ(q) for state."""
        if state is None:
            state = self.current_state
        return self._potentials.get(state, -1.0)

    def step(self, events: Set[str]) -> Tuple[str, float, bool]:
        """
        Process events and return (new_state, reward, is_terminal).

        Reward computation depends on reward_mode:
        - sparse: 1 if terminal else 0
        - distance: -distance(q, F)
        - progression: transition reward
        - hybrid: transition + γ·Φ(q') - Φ(q)

        State validation:
        - Before processing transitions, validate current state
        - If validation fails, regress to fallback state with penalty
        - Potential-based shaping automatically penalizes regression

        Args:
            events: Set of currently true propositions

        Returns:
            Tuple of (new_state, reward, is_terminal)
        """
        # Step 1: Validate current state (may cause regression)
        regression_penalty = 0.0
        regressed = False
        regression_from = None
        regression_to = None

        if self.enable_state_validation:
            regressed, regression_from, regression_to, regression_penalty = \
                self._validate_current_state(events)

        old_state = self.current_state
        old_potential = self._potentials.get(old_state, 0.0)
        transition_reward = 0.0
        transitioned = False

        # Step 2: Check all possible transitions from current state
        for event in events:
            key = (self.current_state, event)
            if key in self.transitions:
                new_state = self.transitions[key]

                # Get transition reward
                reward_key = (self.current_state, new_state)
                if reward_key in self.rewards:
                    transition_reward = self.rewards[reward_key]

                # Update state
                self.current_state = new_state
                transitioned = True

                # Update statistics
                self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
                self._state_visits[new_state] = self._state_visits.get(new_state, 0) + 1

                # Only process first matching transition
                break

        new_potential = self._potentials.get(self.current_state, 0.0)
        is_terminal = self.current_state in self.terminal_states

        # Step 3: Compute reward based on mode
        reward = self._compute_reward(
            old_state=old_state,
            new_state=self.current_state,
            old_potential=old_potential,
            new_potential=new_potential,
            transition_reward=transition_reward,
            transitioned=transitioned,
            is_terminal=is_terminal,
            regressed=regressed,
            regression_penalty=regression_penalty,
        )

        self._total_rm_reward += reward

        return self.current_state, reward, is_terminal

    def _validate_current_state(self, events: Set[str]) -> Tuple[bool, Optional[str], Optional[str], float]:
        """
        Validate current state against required events.

        If current state has a validator and required events are not present,
        regress to the fallback state.

        Args:
            events: Set of currently true propositions

        Returns:
            Tuple of (regressed, from_state, to_state, penalty)
        """
        if self.current_state not in self.state_validators:
            return False, None, None, 0.0

        required_events, fallback_state, penalty = self.state_validators[self.current_state]

        # Check if ALL required events are present
        if not required_events.issubset(events):
            # Validation failed - regress!
            from_state = self.current_state
            to_state = fallback_state

            # Update state
            self.current_state = fallback_state

            # Update statistics
            regression_key = (from_state, to_state)
            self._regression_counts[regression_key] = \
                self._regression_counts.get(regression_key, 0) + 1
            self._total_regressions += 1
            self._total_regression_penalty += penalty
            self._state_visits[to_state] = self._state_visits.get(to_state, 0) + 1

            return True, from_state, to_state, penalty

        return False, None, None, 0.0

    def _compute_reward(
        self,
        old_state: str,
        new_state: str,
        old_potential: float,
        new_potential: float,
        transition_reward: float,
        transitioned: bool,
        is_terminal: bool,
        regressed: bool = False,
        regression_penalty: float = 0.0,
    ) -> float:
        """
        Compute reward based on reward_mode.

        Modes:
        - sparse: R = terminal_reward if is_terminal else 0
        - distance: R = -distance(new_state, F) / max_distance
        - progression: R = transition_reward
        - hybrid: R = transition_reward + potential_scale * (γ·Φ(q') - Φ(q))

        Regression handling:
        - If regressed, apply regression_penalty
        - Potential-based shaping automatically penalizes regression (Φ decreases)
        """
        reward = 0.0

        if self.reward_mode == 'sparse':
            reward = self.terminal_reward if is_terminal else 0.0

        elif self.reward_mode == 'distance':
            # Negative normalized distance
            dist = self._distances.get(new_state, len(self.states))
            max_dist = max(self._distances.values()) if self._distances else 1
            reward = -dist / max_dist

        elif self.reward_mode == 'progression':
            # Only transition reward + terminal bonus
            reward = transition_reward
            if is_terminal:
                reward += self.terminal_reward

        elif self.reward_mode == 'hybrid':
            # Progression + potential-based shaping
            reward = transition_reward

            # Potential-based shaping: F = γ·Φ(s') - Φ(s)
            # This is guaranteed to not change optimal policy (Ng et al. 1999)
            # Note: When regressing, new_potential < old_potential, so shaping is negative
            if self.use_potential_shaping:
                shaping = self.gamma * new_potential - old_potential
                self._total_shaping_reward += shaping * self.potential_scale
                reward += self.potential_scale * shaping

            # Terminal bonus
            if is_terminal:
                reward += self.terminal_reward

        else:
            reward = transition_reward

        # Apply regression penalty (additional to potential shaping)
        if regressed and regression_penalty > 0:
            reward -= regression_penalty

        return reward

    def reset(self, reset_episode_stats: bool = False) -> str:
        """
        Reset RM to initial state.

        Args:
            reset_episode_stats: If True, also reset episode-specific stats
                                 (useful for evaluation)
        """
        self.current_state = self.initial_state
        self._episode_count += 1
        self._state_visits[self.initial_state] = \
            self._state_visits.get(self.initial_state, 0) + 1

        if reset_episode_stats:
            self._total_rm_reward = 0.0
            self._total_shaping_reward = 0.0
            self._total_regression_penalty = 0.0

        return self.current_state

    def get_state_index(self) -> int:
        """Get current state as integer index (for one-hot encoding)."""
        states_list = sorted(self.states)
        return states_list.index(self.current_state)

    def get_state_one_hot(self) -> np.ndarray:
        """Get current state as one-hot vector."""
        n_states = len(self.states)
        one_hot = np.zeros(n_states, dtype=np.float32)
        one_hot[self.get_state_index()] = 1.0
        return one_hot

    @property
    def n_states(self) -> int:
        """Number of states."""
        return len(self.states)

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self.current_state in self.terminal_states

    def get_progress(self) -> float:
        """
        Get task progress as fraction [0, 1] based on distance.

        Progress = 1 - (distance / max_distance)
        """
        if not self._distances:
            return 0.0

        dist = self._distances.get(self.current_state, len(self.states))
        max_dist = max(self._distances.values())

        if max_dist == 0:
            return 1.0

        return 1.0 - (dist / max_dist)

    def get_diagnostics(self) -> OrderedDict:
        """Get RM diagnostics for logging."""
        diag = OrderedDict()
        diag['rm/name'] = self.name
        diag['rm/current_state'] = self.current_state
        diag['rm/state_index'] = self.get_state_index()
        diag['rm/progress'] = self.get_progress()
        diag['rm/distance_to_terminal'] = self.distance_to_terminal()
        diag['rm/potential'] = self.potential()
        diag['rm/is_terminal'] = self.is_terminal()
        diag['rm/total_reward'] = self._total_rm_reward
        diag['rm/total_shaping_reward'] = self._total_shaping_reward
        diag['rm/episode_count'] = self._episode_count
        diag['rm/reward_mode'] = self.reward_mode

        # State validation statistics
        diag['rm/state_validation_enabled'] = self.enable_state_validation
        diag['rm/total_regressions'] = self._total_regressions
        diag['rm/total_regression_penalty'] = self._total_regression_penalty

        # State visit distribution
        total_visits = sum(self._state_visits.values())
        if total_visits > 0:
            for state in sorted(self.states):
                ratio = self._state_visits.get(state, 0) / total_visits
                diag[f'rm/state_ratio/{state}'] = ratio
                diag[f'rm/distance/{state}'] = self._distances.get(state, -1)

        # Regression counts per transition
        for (from_s, to_s), count in self._regression_counts.items():
            diag[f'rm/regression/{from_s}_to_{to_s}'] = count

        return diag

    def __repr__(self) -> str:
        return (f"RewardMachine({self.name}, state={self.current_state}, "
                f"mode={self.reward_mode}, dist={self.distance_to_terminal()}, "
                f"terminal={self.terminal_states})")


class CompositeRewardMachine:
    """
    Composite RM for tasks with multiple parallel sub-tasks (e.g., Cleanup).

    Manages multiple RMs and aggregates their rewards.
    """

    def __init__(
        self,
        reward_machines: List[RewardMachine],
        aggregation: str = 'sum',
        name: str = "CompositeRM"
    ):
        """
        Initialize composite RM.

        Args:
            reward_machines: List of component RMs
            aggregation: How to aggregate rewards ('sum', 'mean', 'max')
            name: Name for logging
        """
        self.reward_machines = reward_machines
        self.aggregation = aggregation
        self.name = name

    def step(self, events: Set[str]) -> Tuple[List[str], float, bool]:
        """
        Process events through all component RMs.

        Returns:
            Tuple of (list of states, aggregated reward, all terminal)
        """
        states = []
        rewards = []
        all_terminal = True

        for rm in self.reward_machines:
            state, reward, is_terminal = rm.step(events)
            states.append(state)
            rewards.append(reward)
            all_terminal = all_terminal and is_terminal

        # Aggregate rewards
        if self.aggregation == 'sum':
            total_reward = sum(rewards)
        elif self.aggregation == 'mean':
            total_reward = float(np.mean(rewards))
        elif self.aggregation == 'max':
            total_reward = max(rewards)
        else:
            total_reward = sum(rewards)

        return states, total_reward, all_terminal

    def reset(self) -> List[str]:
        """Reset all component RMs."""
        return [rm.reset() for rm in self.reward_machines]

    def get_state_one_hot(self) -> np.ndarray:
        """Get concatenated one-hot vectors from all RMs."""
        return np.concatenate([rm.get_state_one_hot() for rm in self.reward_machines])

    @property
    def n_states(self) -> int:
        """Total number of states across all RMs."""
        return sum(rm.n_states for rm in self.reward_machines)

    def is_terminal(self) -> bool:
        """Check if all component RMs are terminal."""
        return all(rm.is_terminal() for rm in self.reward_machines)

    def get_progress(self) -> float:
        """Get average progress across all component RMs."""
        if not self.reward_machines:
            return 0.0
        return float(np.mean([rm.get_progress() for rm in self.reward_machines]))

    def get_diagnostics(self) -> OrderedDict:
        """Get aggregated diagnostics."""
        diag = OrderedDict()
        diag['composite_rm/name'] = self.name
        diag['composite_rm/n_components'] = len(self.reward_machines)
        diag['composite_rm/all_terminal'] = self.is_terminal()
        diag['composite_rm/avg_progress'] = self.get_progress()

        for i, rm in enumerate(self.reward_machines):
            rm_diag = rm.get_diagnostics()
            for key, value in rm_diag.items():
                diag[f'component_{i}/{key}'] = value

        return diag
