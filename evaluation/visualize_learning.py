"""Visualization tools for RL learning progress.

This module provides functions to visualize the learning process
of RL agents.
"""

import argparse
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')


def load_training_logs(log_file: str) -> Dict:
    """Load training logs from file.

    Args:
        log_file: Path to log file

    Returns:
        Dictionary with training data
    """
    with open(log_file, 'rb') as f:
        return pickle.load(f)


def plot_learning_curve(
    episodes: List[int],
    rewards: List[float],
    window: int = 50,
    title: str = 'Learning Curve',
    save_path: Optional[str] = None
) -> None:
    """Plot learning curve with moving average.

    Args:
        episodes: Episode numbers
        rewards: Rewards per episode
        window: Moving average window
        title: Plot title
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label='Episode Reward', color='blue')

    # Moving average
    if len(rewards) >= window:
        ma_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ma_episodes = episodes[window - 1:]
        ax.plot(ma_episodes, ma_rewards, label=f'{window}-Episode MA', linewidth=2, color='red')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()
    plt.close()


def plot_accuracy_over_time(
    episodes: List[int],
    accuracies: List[float],
    window: int = 50,
    save_path: Optional[str] = None
) -> None:
    """Plot accuracy over training episodes.

    Args:
        episodes: Episode numbers
        accuracies: Accuracy per episode
        window: Moving average window
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw accuracy
    ax.plot(episodes, accuracies, alpha=0.3, label='Episode Accuracy', color='green')

    # Moving average
    if len(accuracies) >= window:
        ma_acc = np.convolve(accuracies, np.ones(window) / window, mode='valid')
        ma_episodes = episodes[window - 1:]
        ax.plot(ma_episodes, ma_acc, label=f'{window}-Episode MA', linewidth=2, color='darkgreen')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add horizontal line at random baseline
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()
    plt.close()


def plot_exploration_exploitation(
    episodes: List[int],
    epsilon_values: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plot epsilon decay over time.

    Args:
        episodes: Episode numbers
        epsilon_values: Epsilon values
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, epsilon_values, linewidth=2, color='purple')
    ax.fill_between(episodes, epsilon_values, alpha=0.3, color='purple')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax.set_title('Exploration vs Exploitation', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(epsilon_values) * 1.1])
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(
        'High Exploration',
        xy=(episodes[0], epsilon_values[0]),
        xytext=(episodes[len(episodes) // 4], epsilon_values[0] * 0.8),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        fontsize=10
    )

    if len(episodes) > 10:
        ax.annotate(
            'High Exploitation',
            xy=(episodes[-1], epsilon_values[-1]),
            xytext=(episodes[-len(episodes) // 4], epsilon_values[0] * 0.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()
    plt.close()


def plot_q_value_heatmap(
    q_table: Dict,
    actions: List[str],
    num_states: int = 20,
    save_path: Optional[str] = None
) -> None:
    """Plot heatmap of Q-values.

    Args:
        q_table: Q-table dictionary
        actions: List of action names
        num_states: Number of states to visualize
        save_path: Path to save plot
    """
    # Sample states
    states = list(q_table.keys())[:num_states]

    # Build matrix
    q_matrix = np.zeros((len(states), len(actions)))

    for i, state in enumerate(states):
        for j, action in enumerate(actions):
            q_matrix[i, j] = q_table[state].get(action, 0.0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        q_matrix,
        xticklabels=actions,
        yticklabels=[f'State {i}' for i in range(len(states))],
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.2f',
        ax=ax
    )

    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title('Q-Value Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()
    plt.close()


def plot_training_comparison(
    results: Dict[str, Dict],
    metric: str = 'reward',
    window: int = 50,
    save_path: Optional[str] = None
) -> None:
    """Compare training curves of multiple agents.

    Args:
        results: Dictionary mapping agent names to training data
        metric: Metric to plot ('reward' or 'accuracy')
        window: Moving average window
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        episodes = data.get('episodes', list(range(len(data[metric]))))
        values = data[metric]

        # Moving average
        if len(values) >= window:
            ma_values = np.convolve(values, np.ones(window) / window, mode='valid')
            ma_episodes = episodes[window - 1:]
            ax.plot(ma_episodes, ma_values, label=name, linewidth=2, color=color)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.show()
    plt.close()


def create_training_dashboard(
    rewards: List[float],
    accuracies: List[float],
    epsilon_values: Optional[List[float]] = None,
    losses: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """Create comprehensive training dashboard.

    Args:
        rewards: Episode rewards
        accuracies: Episode accuracies
        epsilon_values: Epsilon values (optional)
        losses: Training losses (optional)
        save_path: Path to save plot
    """
    # Determine grid size
    num_plots = 2  # rewards and accuracy
    if epsilon_values:
        num_plots += 1
    if losses:
        num_plots += 1

    rows = (num_plots + 1) // 2
    cols = 2

    fig = plt.figure(figsize=(14, 5 * rows))

    episodes = np.arange(len(rewards))
    window = 50

    # Plot 1: Rewards
    ax1 = plt.subplot(rows, cols, 1)
    ax1.plot(episodes, rewards, alpha=0.3, color='blue')
    if len(rewards) >= window:
        ma_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(episodes[window - 1:], ma_rewards, linewidth=2, color='darkblue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2 = plt.subplot(rows, cols, 2)
    ax2.plot(episodes, accuracies, alpha=0.3, color='green')
    if len(accuracies) >= window:
        ma_acc = np.convolve(accuracies, np.ones(window) / window, mode='valid')
        ax2.plot(episodes[window - 1:], ma_acc, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_ylim([-0.05, 1.05])
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plot_idx = 3

    # Plot 3: Epsilon (if available)
    if epsilon_values:
        ax3 = plt.subplot(rows, cols, plot_idx)
        ax3.plot(episodes, epsilon_values, linewidth=2, color='purple')
        ax3.fill_between(episodes, epsilon_values, alpha=0.3, color='purple')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate')
        ax3.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 4: Losses (if available)
    if losses:
        ax4 = plt.subplot(rows, cols, plot_idx)
        loss_episodes = np.arange(len(losses))
        ax4.plot(loss_episodes, losses, alpha=0.5, color='orange', marker='o')
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved: {save_path}")

    plt.show()
    plt.close()


def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(description='Visualize RL training progress')
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to training log file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_file:
        # Load and visualize training logs
        logs = load_training_logs(args.log_file)

        # Create dashboard
        create_training_dashboard(
            rewards=logs.get('rewards', []),
            accuracies=logs.get('accuracies', []),
            epsilon_values=logs.get('epsilon_values'),
            losses=logs.get('losses'),
            save_path=os.path.join(args.output_dir, 'training_dashboard.png')
        )
    else:
        print("No log file provided. Use --log-file to specify training logs.")


if __name__ == '__main__':
    main()
