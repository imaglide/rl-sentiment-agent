"""Training script for Policy Gradient agent.

This script trains a policy gradient (REINFORCE) agent on sentiment classification.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.policy_gradient import PolicyGradientAgent
from environments.sentiment_env import SentimentEnv, SimpleTokenizer
from environments.reward_shaping import create_reward_shaper


def train_policy_gradient(
    episodes: int = 500,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    eval_every: int = 25,
    save_dir: str = 'checkpoints',
    dataset: str = 'imdb',
    subset_size: int = 1000,
    vocab_size: int = 5000,
    batch_size: int = 10,
    device: str = None
):
    """Train policy gradient agent.

    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate
        gamma: Discount factor
        eval_every: Evaluation frequency
        save_dir: Directory to save checkpoints
        dataset: Dataset name
        subset_size: Size of data subset
        vocab_size: Vocabulary size
        batch_size: Batch size for updates
        device: Device to use
    """
    print("=" * 60)
    print("Training Policy Gradient Agent")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Batch size: {batch_size}")
    print(f"Dataset: {dataset}")
    print(f"Vocab size: {vocab_size}")
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

    # Build tokenizer
    print("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    # Sample texts for vocabulary
    sample_texts = []
    for i in range(min(subset_size, len(env.dataset))):
        if 'text' in env.dataset[i]:
            sample_texts.append(env.dataset[i]['text'])
        elif 'sentence' in env.dataset[i]:
            sample_texts.append(env.dataset[i]['sentence'])

    tokenizer.build_vocab(sample_texts)

    # Initialize agent
    agent = PolicyGradientAgent(
        vocab_size=tokenizer.get_vocab_size(),
        action_labels=env.get_all_sentiments(),
        learning_rate=learning_rate,
        gamma=gamma,
        device=device
    )

    agent.train_mode()

    # Training metrics
    episode_rewards = []
    episode_accuracies = []
    policy_losses = []

    # Training loop
    print("\nTraining...")
    batch_count = 0

    for episode in tqdm(range(episodes)):
        # Reset environment
        text, true_label = env.reset()

        # Tokenize text
        text_indices = tokenizer.encode(text)

        # Agent selects action
        action_idx, action_label = agent.select_action(text_indices, explore=True)

        # Environment step
        _, reward, done, info = env.step(action_label)

        # Store reward
        agent.store_reward(reward)

        # Track metrics
        episode_rewards.append(reward)
        episode_accuracies.append(int(info['correct']))

        # Update policy every batch_size episodes
        if (episode + 1) % batch_size == 0:
            loss = agent.learn()
            policy_losses.append(loss)
            batch_count += 1

        # Evaluation
        if (episode + 1) % eval_every == 0:
            recent_reward = np.mean(episode_rewards[-eval_every:])
            recent_acc = np.mean(episode_accuracies[-eval_every:])

            if policy_losses:
                recent_loss = np.mean(policy_losses[-max(1, eval_every // batch_size):])
            else:
                recent_loss = 0.0

            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {recent_reward:.3f}")
            print(f"  Accuracy: {recent_acc:.3f}")
            print(f"  Policy Loss: {recent_loss:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                save_dir,
                f'policy_gradient_ep{episode + 1}.pth'
            )
            agent.save(checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

    # Final save
    final_path = os.path.join(save_dir, 'policy_gradient_final.pth')
    agent.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Plot learning curves
    plot_learning_curves(
        episode_rewards,
        episode_accuracies,
        policy_losses,
        save_dir,
        batch_size
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
    losses: list,
    save_dir: str,
    batch_size: int,
    window: int = 50
):
    """Plot and save learning curves.

    Args:
        rewards: Episode rewards
        accuracies: Episode accuracies
        losses: Policy losses
        save_dir: Directory to save plots
        batch_size: Batch size used for training
        window: Moving average window
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Compute moving averages
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')

    episodes = np.arange(len(rewards))

    # Rewards
    axes[0].plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    ma_rewards = moving_average(rewards, window)
    axes[0].plot(
        episodes[len(rewards) - len(ma_rewards):],
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
    ma_acc = moving_average(accuracies, window)
    axes[1].plot(
        episodes[len(accuracies) - len(ma_acc):],
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

    # Policy Loss
    if losses:
        batch_episodes = np.arange(0, len(rewards), batch_size)[:len(losses)]
        axes[2].plot(batch_episodes, losses, alpha=0.5, label='Policy Loss', marker='o')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Policy Gradient Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'policy_gradient_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nLearning curves saved: {plot_path}")

    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train Policy Gradient agent for sentiment classification'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for policy updates'
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=25,
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
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=5000,
        help='Vocabulary size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    # Train
    train_policy_gradient(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        dataset=args.dataset,
        subset_size=args.subset_size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
