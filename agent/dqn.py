"""Deep Q-Network (DQN) agent for sentiment classification.

This module implements a DQN agent with experience replay and
target networks for stable learning.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque, namedtuple
import numpy as np


# Experience tuple
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done']
)


class QNetwork(nn.Module):
    """Q-Network that estimates Q-values for each action.

    Architecture:
        - Embedding layer
        - LSTM for sequence processing
        - Linear layers to output Q-values for each action
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_actions: int = 3,
        dropout: float = 0.3
    ):
        """Initialize the Q-network.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_actions: Number of possible actions
            dropout: Dropout probability
        """
        super(QNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            text_indices: Tensor of word indices (batch_size, seq_len)

        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        # Embed tokens
        embedded = self.embedding(text_indices)  # (batch, seq, embed)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden)

        # Apply dropout
        dropped = self.dropout(last_hidden)

        # Hidden layer with ReLU
        x = F.relu(self.fc1(dropped))

        # Q-values for each action
        q_values = self.fc2(x)  # (batch, num_actions)

        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN.

    Stores experiences and samples random batches for training.
    """

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool
    ) -> None:
        """Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
            done: Whether episode ended
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for sentiment classification.

    Implements DQN with:
    - Experience replay for breaking correlations
    - Target network for stable Q-learning
    - Epsilon-greedy exploration

    Q-learning update:
    Q(s,a) = r + γ * max_a' Q_target(s', a')
    """

    def __init__(
        self,
        vocab_size: int,
        action_labels: Optional[List[str]] = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        device: Optional[str] = None
    ):
        """Initialize the DQN agent.

        Args:
            vocab_size: Size of vocabulary
            action_labels: List of action labels
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to use ('cuda' or 'cpu')
        """
        self.action_labels = action_labels or ['positive', 'negative', 'neutral']
        self.num_actions = len(self.action_labels)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Q-networks (online and target)
        self.q_network = QNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_network = QNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_actions=self.num_actions
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.update_count = 0

    def select_action(
        self,
        text_indices: torch.Tensor,
        explore: bool = True
    ) -> Tuple[int, str]:
        """Select an action using epsilon-greedy policy.

        Args:
            text_indices: Tokenized text as tensor
            explore: Whether to use epsilon-greedy exploration

        Returns:
            Tuple of (action_index, action_label)
        """
        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            # EXPLORE: Random action
            action_idx = random.randrange(self.num_actions)
        else:
            # EXPLOIT: Best action according to Q-network
            if text_indices.dim() == 1:
                text_indices = text_indices.unsqueeze(0)

            text_indices = text_indices.to(self.device)

            with torch.no_grad():
                q_values = self.q_network(text_indices)
                action_idx = q_values.argmax(dim=-1).item()

        action_label = self.action_labels[action_idx]
        return action_idx, action_label

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool
    ) -> None:
        """Store an experience in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def learn(self) -> Optional[float]:
        """Update Q-network using a batch from replay buffer.

        Returns:
            Loss value if update occurred, None otherwise
        """
        # Need enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare batch tensors
        states = torch.stack([exp.state.squeeze() for exp in batch]).to(self.device)
        actions = torch.tensor(
            [exp.action for exp in batch],
            dtype=torch.long,
            device=self.device
        )
        rewards = torch.tensor(
            [exp.reward for exp in batch],
            dtype=torch.float,
            device=self.device
        )

        # Handle next states (some may be None for terminal states)
        non_final_mask = torch.tensor(
            [exp.next_state is not None for exp in batch],
            dtype=torch.bool,
            device=self.device
        )
        non_final_next_states = torch.stack(
            [exp.next_state.squeeze() for exp in batch if exp.next_state is not None]
        ).to(self.device) if any(non_final_mask) else None

        # Compute Q(s, a)
        q_values = self.q_network(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute max Q(s', a') using target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states is not None:
            with torch.no_grad():
                next_state_values[non_final_mask] = (
                    self.target_network(non_final_next_states).max(1)[0]
                )

        # Compute target Q-values
        expected_state_action_values = rewards + self.gamma * next_state_values

        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.update_count += 1

        # Update target network periodically
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_episodes += 1

    def save(self, filepath: str) -> None:
        """Save the agent to disk.

        Args:
            filepath: Path to save the agent
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_labels': self.action_labels,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        vocab_size: int,
        device: Optional[str] = None
    ) -> 'DQNAgent':
        """Load a saved agent from disk.

        Args:
            filepath: Path to the saved agent
            vocab_size: Vocabulary size
            device: Device to load on

        Returns:
            Loaded DQNAgent instance
        """
        checkpoint = torch.load(filepath, map_location=device)

        agent = cls(
            vocab_size=vocab_size,
            action_labels=checkpoint['action_labels'],
            gamma=checkpoint['gamma'],
            epsilon=checkpoint['epsilon'],
            epsilon_decay=checkpoint['epsilon_decay'],
            epsilon_min=checkpoint['epsilon_min'],
            device=device
        )

        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.total_episodes = checkpoint['total_episodes']
        agent.total_steps = checkpoint['total_steps']
        agent.update_count = checkpoint['update_count']

        return agent

    def get_stats(self) -> Dict[str, any]:
        """Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.q_network.parameters())
        }
