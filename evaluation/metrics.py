"""Evaluation metrics for RL sentiment agents.

This module provides functions to evaluate trained agents.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.q_learning import QLearningAgent
from agent.policy_gradient import PolicyGradientAgent
from agent.dqn import DQNAgent
from environments.sentiment_env import SentimentEnv, SimpleTokenizer


def evaluate_agent(
    agent,
    env: SentimentEnv,
    tokenizer: Optional[SimpleTokenizer] = None,
    num_episodes: int = 100,
    agent_type: str = 'q_learning'
) -> Dict:
    """Evaluate an agent on the environment.

    Args:
        agent: Trained agent
        env: Evaluation environment
        tokenizer: Tokenizer (for neural agents)
        num_episodes: Number of evaluation episodes
        agent_type: Type of agent ('q_learning', 'policy_gradient', 'dqn')

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating agent for {num_episodes} episodes...")

    predictions = []
    true_labels = []
    rewards = []

    for _ in tqdm(range(num_episodes)):
        # Reset environment
        text, true_label = env.reset()

        # Get prediction
        if agent_type == 'q_learning':
            prediction = agent.predict(text, explore=False)
        elif agent_type in ['policy_gradient', 'dqn']:
            if tokenizer is None:
                raise ValueError("Tokenizer required for neural agents")

            agent.eval_mode() if hasattr(agent, 'eval_mode') else None
            text_indices = tokenizer.encode(text)

            with torch.no_grad():
                action_idx, prediction = agent.select_action(
                    text_indices,
                    explore=False
                )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Step environment
        _, reward, done, info = env.step(prediction)

        # Store results
        predictions.append(info['predicted_label'])
        true_labels.append(info['true_label'])
        rewards.append(reward)

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average=None,
            zero_division=0
        )
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'support': support,
        'confusion_matrix': conf_matrix,
        'predictions': predictions,
        'true_labels': true_labels
    }

    return results


def print_results(results: Dict, label_names: List[str]) -> None:
    """Print evaluation results.

    Args:
        results: Results dictionary from evaluate_agent
        label_names: Names of labels
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")

    print(f"\nPer-Class Metrics:")
    for i, label in enumerate(label_names):
        if i < len(results['per_class_f1']):
            print(f"  {label}:")
            print(f"    Precision: {results['per_class_precision'][i]:.4f}")
            print(f"    Recall:    {results['per_class_recall'][i]:.4f}")
            print(f"    F1 Score:  {results['per_class_f1'][i]:.4f}")
            print(f"    Support:   {results['support'][i]}")

    print("\n" + "=" * 60)


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        label_names: Names of labels
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved: {save_path}")

    plt.show()
    plt.close()


def compare_agents(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """Compare multiple agents.

    Args:
        results_dict: Dictionary mapping agent names to results
        save_path: Path to save comparison plot
    """
    agents = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results_dict[agent][metric] for agent in agents]

        axes[i].bar(agents, values, alpha=0.7)
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_ylim([0, 1])
        axes[i].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved: {save_path}")

    plt.show()
    plt.close()


def main():
    """Main function for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate RL sentiment agent')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to agent checkpoint'
    )
    parser.add_argument(
        '--agent-type',
        type=str,
        choices=['q_learning', 'policy_gradient', 'dqn'],
        required=True,
        help='Type of agent'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
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
        default=500,
        help='Size of evaluation subset'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=5000,
        help='Vocabulary size (for neural agents)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load environment
    env = SentimentEnv(
        dataset_name=args.dataset,
        split='test',
        use_subset=True,
        subset_size=args.subset_size
    )

    label_names = env.get_all_sentiments()

    # Load agent
    tokenizer = None

    if args.agent_type == 'q_learning':
        agent = QLearningAgent.load(args.checkpoint)

    elif args.agent_type == 'policy_gradient':
        # Build tokenizer
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
        sample_texts = [
            env.dataset[i].get('text', env.dataset[i].get('sentence', ''))
            for i in range(min(args.subset_size, len(env.dataset)))
        ]
        tokenizer.build_vocab(sample_texts)

        agent = PolicyGradientAgent.load(
            args.checkpoint,
            vocab_size=tokenizer.get_vocab_size()
        )

    elif args.agent_type == 'dqn':
        # Build tokenizer
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
        sample_texts = [
            env.dataset[i].get('text', env.dataset[i].get('sentence', ''))
            for i in range(min(args.subset_size, len(env.dataset)))
        ]
        tokenizer.build_vocab(sample_texts)

        agent = DQNAgent.load(
            args.checkpoint,
            vocab_size=tokenizer.get_vocab_size()
        )

    # Evaluate
    results = evaluate_agent(
        agent,
        env,
        tokenizer=tokenizer,
        num_episodes=args.num_episodes,
        agent_type=args.agent_type
    )

    # Print results
    print_results(results, label_names)

    # Plot confusion matrix
    conf_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], label_names, conf_matrix_path)


if __name__ == '__main__':
    main()
