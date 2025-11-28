#!/usr/bin/env python3
"""Quick demo of RL sentiment agent.

This script provides a simple demonstration of training and using
a Q-learning agent for sentiment classification.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.q_learning import QLearningAgent
from environments.sentiment_env import SentimentEnv
import numpy as np


def main():
    """Run demo."""
    print("=" * 60)
    print("RL SENTIMENT AGENT - Quick Demo")
    print("=" * 60)

    # Initialize environment
    print("\n1. Initializing environment...")
    env = SentimentEnv(
        dataset_name='imdb',
        split='train',
        use_subset=True,
        subset_size=100  # Small subset for quick demo
    )
    print(f"   Loaded {len(env.dataset)} samples")

    # Initialize agent
    print("\n2. Creating Q-Learning agent...")
    agent = QLearningAgent(
        actions=env.get_all_sentiments(),
        alpha=0.1,
        gamma=0.9,
        epsilon=0.5,
        epsilon_decay=0.99
    )
    print(f"   Actions: {agent.actions}")

    # Train
    print("\n3. Training agent (50 episodes)...")
    num_episodes = 50
    episode_rewards = []
    episode_accuracies = []

    for episode in range(num_episodes):
        # Reset environment
        text, true_label = env.reset()

        # Agent predicts
        prediction = agent.predict(text, explore=True)

        # Environment step
        _, reward, done, info = env.step(prediction)

        # Agent learns
        agent.learn(text, prediction, reward, done=True)

        # Decay epsilon
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(reward)
        episode_accuracies.append(int(info['correct']))

        if (episode + 1) % 10 == 0:
            recent_acc = np.mean(episode_accuracies[-10:])
            print(f"   Episode {episode + 1}: Accuracy = {recent_acc:.2f}")

    # Final stats
    print("\n4. Training Results:")
    final_accuracy = np.mean(episode_accuracies[-10:])
    print(f"   Final Accuracy (last 10 episodes): {final_accuracy:.2%}")
    print(f"   Q-table size: {len(agent.Q)} states")
    print(f"   Final epsilon: {agent.epsilon:.3f}")

    # Test on examples
    print("\n5. Testing on example texts:")
    test_examples = [
        "This movie was absolutely amazing! Best film I've seen all year.",
        "Terrible movie. Complete waste of time and money.",
        "The film had some good moments but overall was disappointing."
    ]

    for i, text in enumerate(test_examples, 1):
        prediction = agent.predict(text, explore=False)
        q_values = agent.get_state_action_values(text)

        print(f"\n   Example {i}:")
        print(f"   Text: {text[:60]}...")
        print(f"   Prediction: {prediction}")
        print(f"   Q-values: {', '.join(f'{k}={v:.2f}' for k, v in q_values.items())}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Train longer: python -m training.train_q_learning --episodes 1000")
    print("  - Try policy gradient: python -m training.train_pg --episodes 500")
    print("  - Evaluate agent: python -m evaluation.metrics --checkpoint <path>")
    print("  - Run tests: pytest tests/")
    print("=" * 60)


if __name__ == '__main__':
    main()
