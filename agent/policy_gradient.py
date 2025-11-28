"""Policy Gradient agent for sentiment classification.

This module implements a neural network-based policy gradient agent
using the REINFORCE algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import numpy as np


class PolicyNetwork(nn.Module):
    """Neural network that outputs action probabilities.

    Architecture:
        - Embedding layer
        - LSTM for sequence processing
        - Linear layer to output action probabilities
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_actions: int = 3,
        dropout: float = 0.3
    ):
        """Initialize the policy network.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_actions: Number of possible actions
            dropout: Dropout probability
        """
        super(PolicyNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            text_indices: Tensor of word indices (batch_size, seq_len)

        Returns:
            Action probabilities (batch_size, num_actions)
        """
        # Embed tokens
        embedded = self.embedding(text_indices)  # (batch, seq, embed)

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden)

        # Apply dropout
        dropped = self.dropout(last_hidden)

        # Output layer
        logits = self.fc(dropped)  # (batch, num_actions)

        # Softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)

        return action_probs


class PolicyGradientAgent:
    """REINFORCE policy gradient agent for sentiment classification.

    The agent learns a parameterized policy π(a|s; θ) and updates
    parameters using the policy gradient theorem:

    ∇J(θ) = E[∇log π(a|s; θ) * R]

    where R is the return (cumulative reward).
    """

    def __init__(
        self,
        vocab_size: int,
        action_labels: Optional[List[str]] = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: Optional[str] = None
    ):
        """Initialize the policy gradient agent.

        Args:
            vocab_size: Size of vocabulary
            action_labels: List of action labels
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for returns
            device: Device to use ('cuda' or 'cpu')
        """
        self.action_labels = action_labels or ['positive', 'negative', 'neutral']
        self.num_actions = len(self.action_labels)
        self.gamma = gamma

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Policy network
        self.policy = PolicyNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_actions=self.num_actions
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Episode buffers
        self.saved_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    def select_action(
        self,
        text_indices: torch.Tensor,
        explore: bool = True
    ) -> Tuple[int, str]:
        """Select an action using the current policy.

        Args:
            text_indices: Tokenized text as tensor
            explore: Whether to sample (True) or take argmax (False)

        Returns:
            Tuple of (action_index, action_label)
        """
        # Ensure correct shape (add batch dimension if needed)
        if text_indices.dim() == 1:
            text_indices = text_indices.unsqueeze(0)

        text_indices = text_indices.to(self.device)

        # Get action probabilities
        with torch.no_grad() if not self.training else torch.enable_grad():
            action_probs = self.policy(text_indices)

        # Sample or take argmax
        if explore:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            # Save log probability for training
            if self.training:
                self.saved_log_probs.append(dist.log_prob(action))
        else:
            # Greedy: take best action
            action = action_probs.argmax(dim=-1)

        action_idx = action.item()
        action_label = self.action_labels[action_idx]

        return action_idx, action_label

    def store_reward(self, reward: float) -> None:
        """Store a reward for the current episode.

        Args:
            reward: Reward value
        """
        self.rewards.append(reward)
        self.total_steps += 1

    def learn(self) -> float:
        """Update policy using REINFORCE algorithm.

        Computes returns and updates policy parameters using
        the policy gradient.

        Returns:
            Policy loss value
        """
        if len(self.rewards) == 0:
            return 0.0

        # Compute returns (discounted cumulative rewards)
        returns = self._compute_returns()

        # Normalize returns (reduces variance)
        returns = torch.tensor(returns, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            # Policy gradient: -log π(a|s) * R
            policy_loss.append(-log_prob * R)

        # Optimize
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clear episode buffers
        loss_value = loss.item()
        self.clear_episode()

        return loss_value

    def _compute_returns(self) -> List[float]:
        """Compute discounted returns for the episode.

        Returns:
            List of returns for each timestep
        """
        returns = []
        R = 0.0

        # Compute returns backwards
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        return returns

    def clear_episode(self) -> None:
        """Clear episode buffers."""
        self.saved_log_probs.clear()
        self.rewards.clear()
        self.total_episodes += 1

    def train_mode(self) -> None:
        """Set agent to training mode."""
        self.policy.train()
        self.training = True

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.policy.eval()
        self.training = False

    def save(self, filepath: str) -> None:
        """Save the agent to disk.

        Args:
            filepath: Path to save the agent
        """
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_labels': self.action_labels,
            'gamma': self.gamma,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        vocab_size: int,
        device: Optional[str] = None
    ) -> 'PolicyGradientAgent':
        """Load a saved agent from disk.

        Args:
            filepath: Path to the saved agent
            vocab_size: Vocabulary size
            device: Device to load on

        Returns:
            Loaded PolicyGradientAgent instance
        """
        checkpoint = torch.load(filepath, map_location=device)

        agent = cls(
            vocab_size=vocab_size,
            action_labels=checkpoint['action_labels'],
            gamma=checkpoint['gamma'],
            device=device
        )

        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            'gamma': self.gamma,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.policy.parameters())
        }
