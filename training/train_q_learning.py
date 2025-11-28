"""Training script for Q-Learning agent.

This script trains a tabular Q-learning agent on sentiment classification.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.q_learning import QLearningAgent
from environments.sentiment_env import SentimentEnv
from environments.reward_shaping import create_reward_shaper


def train_q_learning(
    episodes: int = 1000,
    learning_rate: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.995,
    reward_shaping: str = 'none',
    eval_every: int = 50,
    save_dir: str = 'checkpoints',
    dataset: str = 'imdb',
    subset_size: int = 1000
):
    """Train Q-learning agent.

    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate (alpha)
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        reward_shaping: Reward shaping strategy
        eval_every: Evaluation frequency
        save_dir: Directory to save checkpoints
        dataset: Dataset name
        subset_size: Size of data subset to use
    """
    print("=" * 60)
    print("Training Q-Learning Agent")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon} -> {epsilon * (epsilon_decay ** episodes):.4f}")
    print(f"Reward shaping: {reward_shaping}")
    print(f"Dataset: {dataset}")
    print("=" * 60)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize environment
    env = SentimentEnv(
        dataset_name=dataset,
        split='train',
        use_subset=True,
        subset_size=subset_size
    )

    # Initialize agent
    agent = QLearningAgent(
        actions=env.get_all_sentiments(),
        alpha=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )

    # Reward shaper
    reward_shaper = create_reward_shaper(reward_shaping)

    # Training metrics
    episode_rewards = []
    episode_accuracies = []
    td_errors = []

    # Training loop
    print("\nTraining...")
    for episode in tqdm(range(episodes)):
        # Reset environment
        text, true_label = env.reset()

        # Agent predicts
        prediction = agent.predict(text, explore=True)

        # Environment step
        _, reward, done, info = env.step(prediction)

        # Shape reward
        shaped_reward = reward_shaper.shape_reward(reward, info)

        # Agent learns
        td_error = agent.learn(text, prediction, shaped_reward, done=done)
        td_errors.append(td_error)

        # Decay epsilon
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(reward)
        episode_accuracies.append(int(info['correct']))

        # Evaluation
        if (episode + 1) % eval_every == 0:
            recent_reward = np.mean(episode_rewards[-eval_every:])
            recent_acc = np.mean(episode_accuracies[-eval_every:])
            recent_td = np.mean(td_errors[-eval_every:])

            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {recent_reward:.3f}")
            print(f"  Accuracy: {recent_acc:.3f}")
            print(f"  Avg TD Error: {recent_td:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table size: {len(agent.Q)}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                save_dir,
                f'q_learning_ep{episode + 1}.pkl'
            )
            agent.save(checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

    # Final save
    final_path = os.path.join(save_dir, 'q_learning_final.pkl')
    agent.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Plot learning curves
    plot_learning_curves(
        episode_rewards,
        episode_accuracies,
        td_errors,
        save_dir
    )

    # Final statistics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 60)

    return agent, episode_rewards, episode_accuracies


def plot_learning_curves(
    rewards: list,
    accuracies: list,
    td_errors: list,
    save_dir: str,
    window: int = 50
):
    """Plot and save learning curves.

    Args:
        rewards: Episode rewards
        accuracies: Episode accuracies
        td_errors: TD errors
        save_dir: Directory to save plots
        window: Moving average window
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Compute moving averages
    def moving_average(data, window):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    episodes = np.arange(len(rewards))

    # Rewards
    axes[0].plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= window:
        ma_rewards = moving_average(rewards, window)
        axes[0].plot(
            episodes[window - 1:],
            ma_rewards,
            label=f'{window}-Episode MA',
            linewidth=2
        )
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(episodes, accuracies, alpha=0.3, label='Episode Accuracy')
    if len(accuracies) >= window:
        ma_acc = moving_average(accuracies, window)
        axes[1].plot(
            episodes[window - 1:],
            ma_acc,
            label=f'{window}-Episode MA',
            linewidth=2
        )
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Episode Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    # TD Error
    axes[2].plot(episodes, td_errors, alpha=0.3, label='TD Error')
    if len(td_errors) >= window:
        ma_td = moving_average(td_errors, window)
        axes[2].plot(
            episodes[window - 1:],
            ma_td,
            label=f'{window}-Episode MA',
            linewidth=2
        )
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('TD Error')
    axes[2].set_title('Temporal Difference Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'q_learning_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nLearning curves saved: {plot_path}")

    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train Q-Learning agent for sentiment classification'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate (alpha)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='Discount factor'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.3,
        help='Initial exploration rate'
    )
    parser.add_argument(
        '--epsilon-decay',
        type=float,
        default=0.995,
        help='Epsilon decay rate'
    )
    parser.add_argument(
        '--reward-shaping',
        type=str,
        default='none',
        choices=['none', 'confidence', 'curriculum', 'dense', 'adaptive'],
        help='Reward shaping strategy'
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=50,
        help='Evaluation frequency'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='imdb',
        help='Dataset name'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=1000,
        help='Size of data subset to use'
    )

    args = parser.parse_args()

    # Train
    train_q_learning(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        reward_shaping=args.reward_shaping,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        dataset=args.dataset,
        subset_size=args.subset_size
    )


if __name__ == '__main__':
    main()
