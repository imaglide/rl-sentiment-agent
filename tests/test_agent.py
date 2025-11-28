"""Unit tests for RL agents."""

import unittest
import tempfile
import os
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.q_learning import QLearningAgent
from agent.policy_gradient import PolicyGradientAgent
from agent.dqn import DQNAgent


class TestQLearningAgent(unittest.TestCase):
    """Test Q-Learning agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = QLearningAgent(
            actions=['positive', 'negative', 'neutral'],
            alpha=0.1,
            gamma=0.9,
            epsilon=0.3
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.actions), 3)
        self.assertEqual(self.agent.alpha, 0.1)
        self.assertEqual(self.agent.gamma, 0.9)
        self.assertEqual(self.agent.epsilon, 0.3)

    def test_predict(self):
        """Test action prediction."""
        text = "This is a test sentence"
        action = self.agent.predict(text, explore=False)
        self.assertIn(action, self.agent.actions)

    def test_learn(self):
        """Test learning update."""
        text = "Test sentence"
        action = "positive"
        reward = 1.0

        initial_q = self.agent.Q[self.agent._encode_state(text)][action]
        self.agent.learn(text, action, reward, done=True)
        updated_q = self.agent.Q[self.agent._encode_state(text)][action]

        # Q-value should change after learning
        self.assertNotEqual(initial_q, updated_q)

    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

    def test_save_load(self):
        """Test save and load functionality."""
        # Train agent a bit
        self.agent.learn("test", "positive", 1.0, done=True)

        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            self.agent.save(temp_path)

            # Load
            loaded_agent = QLearningAgent.load(temp_path)

            # Check attributes
            self.assertEqual(loaded_agent.actions, self.agent.actions)
            self.assertEqual(loaded_agent.alpha, self.agent.alpha)
            self.assertEqual(loaded_agent.gamma, self.agent.gamma)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestPolicyGradientAgent(unittest.TestCase):
    """Test Policy Gradient agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.agent = PolicyGradientAgent(
            vocab_size=self.vocab_size,
            action_labels=['positive', 'negative', 'neutral'],
            embedding_dim=32,
            hidden_dim=64,
            learning_rate=0.001,
            device='cpu'
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.action_labels), 3)
        self.assertEqual(self.agent.num_actions, 3)
        self.assertEqual(self.agent.device, torch.device('cpu'))

    def test_select_action(self):
        """Test action selection."""
        # Create dummy input
        text_indices = torch.randint(0, self.vocab_size, (10,))

        self.agent.train_mode()
        action_idx, action_label = self.agent.select_action(text_indices, explore=True)

        self.assertIn(action_idx, range(self.agent.num_actions))
        self.assertIn(action_label, self.agent.action_labels)

    def test_store_reward(self):
        """Test reward storage."""
        self.agent.store_reward(1.0)
        self.assertEqual(len(self.agent.rewards), 1)
        self.assertEqual(self.agent.rewards[0], 1.0)

    def test_learn(self):
        """Test learning update."""
        self.agent.train_mode()

        # Create dummy episode
        text_indices = torch.randint(0, self.vocab_size, (10,))

        for _ in range(5):
            self.agent.select_action(text_indices, explore=True)
            self.agent.store_reward(1.0)

        # Learn
        loss = self.agent.learn()

        # Check buffers are cleared
        self.assertEqual(len(self.agent.rewards), 0)
        self.assertEqual(len(self.agent.saved_log_probs), 0)

    def test_save_load(self):
        """Test save and load functionality."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name

        try:
            self.agent.save(temp_path)

            # Load
            loaded_agent = PolicyGradientAgent.load(
                temp_path,
                vocab_size=self.vocab_size,
                device='cpu'
            )

            # Check attributes
            self.assertEqual(loaded_agent.action_labels, self.agent.action_labels)
            self.assertEqual(loaded_agent.gamma, self.agent.gamma)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestDQNAgent(unittest.TestCase):
    """Test DQN agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.agent = DQNAgent(
            vocab_size=self.vocab_size,
            action_labels=['positive', 'negative', 'neutral'],
            embedding_dim=32,
            hidden_dim=64,
            learning_rate=0.001,
            batch_size=4,
            device='cpu'
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.action_labels), 3)
        self.assertEqual(self.agent.num_actions, 3)
        self.assertEqual(self.agent.batch_size, 4)

    def test_select_action(self):
        """Test action selection."""
        text_indices = torch.randint(0, self.vocab_size, (10,))

        action_idx, action_label = self.agent.select_action(
            text_indices,
            explore=False
        )

        self.assertIn(action_idx, range(self.agent.num_actions))
        self.assertIn(action_label, self.agent.action_labels)

    def test_store_experience(self):
        """Test experience storage."""
        state = torch.randint(0, self.vocab_size, (10,))
        action = 0
        reward = 1.0
        next_state = torch.randint(0, self.vocab_size, (10,))

        self.agent.store_experience(state, action, reward, next_state, False)

        self.assertEqual(len(self.agent.replay_buffer), 1)

    def test_learn(self):
        """Test learning with replay buffer."""
        # Fill replay buffer
        for _ in range(10):
            state = torch.randint(0, self.vocab_size, (10,))
            action = np.random.randint(0, self.agent.num_actions)
            reward = 1.0
            next_state = torch.randint(0, self.vocab_size, (10,))

            self.agent.store_experience(state, action, reward, next_state, False)

        # Learn
        loss = self.agent.learn()

        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)

    def test_target_network_update(self):
        """Test target network update."""
        # Get initial target network params
        initial_params = [
            p.clone() for p in self.agent.target_network.parameters()
        ]

        # Update Q-network
        for _ in range(10):
            state = torch.randint(0, self.vocab_size, (10,))
            self.agent.store_experience(state, 0, 1.0, state, False)

        self.agent.learn()

        # Update target network
        self.agent.update_target_network()

        # Target params should now match Q-network
        for target_param, q_param in zip(
            self.agent.target_network.parameters(),
            self.agent.q_network.parameters()
        ):
            self.assertTrue(torch.allclose(target_param, q_param))

    def test_save_load(self):
        """Test save and load functionality."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name

        try:
            self.agent.save(temp_path)

            # Load
            loaded_agent = DQNAgent.load(
                temp_path,
                vocab_size=self.vocab_size,
                device='cpu'
            )

            # Check attributes
            self.assertEqual(loaded_agent.action_labels, self.agent.action_labels)
            self.assertEqual(loaded_agent.gamma, self.agent.gamma)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
