"""Q-Learning agent for sentiment classification.

This module implements a tabular Q-learning agent that learns
to classify sentiment through reinforcement learning.
"""

import random
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class QLearningAgent:
    """Q-Learning agent for sentiment classification.

    The agent maintains a Q-table that maps (state, action) pairs to
    expected rewards. It learns through the Bellman equation:

    Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]

    Attributes:
        actions: List of possible sentiment labels
        Q: Q-table mapping (state, action) to Q-values
        alpha: Learning rate (0 < alpha <= 1)
        gamma: Discount factor (0 <= gamma <= 1)
        epsilon: Exploration rate (0 <= epsilon <= 1)
        epsilon_decay: Decay rate for epsilon
        epsilon_min: Minimum epsilon value
    """

    def __init__(
        self,
        actions: Optional[List[str]] = None,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        state_bins: int = 10000
    ):
        """Initialize the Q-learning agent.

        Args:
            actions: List of possible actions (sentiment labels)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
            state_bins: Number of discrete state bins for hashing
        """
        self.actions = actions or ['positive', 'negative', 'neutral']
        self.Q: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_bins = state_bins

        # Training statistics
        self.total_episodes = 0
        self.total_steps = 0

    def _encode_state(self, text: str, features: Optional[np.ndarray] = None) -> int:
        """Convert text to discrete state representation.

        For tabular Q-learning, we need discrete states. This function
        hashes the text (or features) into a fixed number of bins.

        Args:
            text: Input text
            features: Optional pre-computed feature vector

        Returns:
            Integer state ID
        """
        if features is not None:
            # Use feature vector if provided
            return int(hash(tuple(features)) % self.state_bins)
        else:
            # Simple text hashing
            return int(hash(text) % self.state_bins)

    def predict(
        self,
        text: str,
        features: Optional[np.ndarray] = None,
        explore: bool = True
    ) -> str:
        """Choose an action using epsilon-greedy policy.

        Args:
            text: Input text
            features: Optional feature representation
            explore: Whether to use exploration (epsilon-greedy)

        Returns:
            Chosen action (sentiment label)
        """
        state = self._encode_state(text, features)

        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            # EXPLORE: Random action
            return random.choice(self.actions)
        else:
            # EXPLOIT: Best known action
            q_values = self.Q[state]
            max_q = max(q_values.values())
            # Handle ties randomly
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def learn(
        self,
        text: str,
        action: str,
        reward: float,
        next_text: Optional[str] = None,
        features: Optional[np.ndarray] = None,
        next_features: Optional[np.ndarray] = None,
        done: bool = True
    ) -> float:
        """Update Q-values using the Bellman equation.

        Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]

        Args:
            text: Current state (text)
            action: Action taken
            reward: Reward received
            next_text: Next state text (optional for episodic tasks)
            features: Current state features
            next_features: Next state features
            done: Whether episode is finished

        Returns:
            TD error (temporal difference error)
        """
        state = self._encode_state(text, features)

        # Current Q-value
        old_q = self.Q[state][action]

        # Best next Q-value
        if done or next_text is None:
            next_max_q = 0.0
        else:
            next_state = self._encode_state(next_text, next_features)
            next_max_q = max(self.Q[next_state].values())

        # Q-learning update (Bellman equation)
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.Q[state][action] = new_q

        # TD error for monitoring
        td_error = abs(new_q - old_q)
        self.total_steps += 1

        return td_error

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_episodes += 1

    def get_state_action_values(
        self,
        text: str,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Get Q-values for all actions in a given state.

        Args:
            text: Input text
            features: Optional feature representation

        Returns:
            Dictionary mapping actions to Q-values
        """
        state = self._encode_state(text, features)
        return self.Q[state].copy()

    def save(self, filepath: str) -> None:
        """Save the Q-table and hyperparameters to disk.

        Args:
            filepath: Path to save the agent
        """
        checkpoint = {
            'Q': dict(self.Q),  # Convert defaultdict to regular dict
            'actions': self.actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'state_bins': self.state_bins,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    @classmethod
    def load(cls, filepath: str) -> 'QLearningAgent':
        """Load a saved agent from disk.

        Args:
            filepath: Path to the saved agent

        Returns:
            Loaded QLearningAgent instance
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        agent = cls(
            actions=checkpoint['actions'],
            alpha=checkpoint['alpha'],
            gamma=checkpoint['gamma'],
            epsilon=checkpoint['epsilon'],
            epsilon_decay=checkpoint['epsilon_decay'],
            epsilon_min=checkpoint['epsilon_min'],
            state_bins=checkpoint['state_bins']
        )

        # Restore Q-table
        agent.Q = defaultdict(
            lambda: {action: 0.0 for action in agent.actions},
            checkpoint['Q']
        )
        agent.total_episodes = checkpoint['total_episodes']
        agent.total_steps = checkpoint['total_steps']

        return agent

    def get_stats(self) -> Dict[str, any]:
        """Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'q_table_size': len(self.Q),
            'alpha': self.alpha,
            'gamma': self.gamma
        }
