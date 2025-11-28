"""Sentiment classification environment for RL agents.

This module provides an environment that wraps a sentiment dataset
and provides RL-style interactions.
"""

import random
import torch
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SentimentEnv:
    """Environment for sentiment classification using RL.

    The environment:
    - Presents text samples from a dataset
    - Receives sentiment predictions from the agent
    - Returns rewards based on correctness
    - Tracks episode statistics
    """

    def __init__(
        self,
        dataset_name: str = 'imdb',
        split: str = 'train',
        max_episodes: Optional[int] = None,
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
        max_seq_length: int = 256,
        use_subset: bool = True,
        subset_size: int = 1000
    ):
        """Initialize the sentiment environment.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            max_episodes: Maximum number of episodes (None for unlimited)
            reward_correct: Reward for correct prediction
            reward_incorrect: Reward for incorrect prediction
            max_seq_length: Maximum sequence length
            use_subset: Whether to use a subset of data
            subset_size: Size of subset if use_subset is True
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_episodes = max_episodes
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.max_seq_length = max_seq_length

        # Load dataset
        print(f"Loading dataset: {dataset_name} ({split})...")
        dataset = load_dataset(dataset_name, split=split)

        # Use subset if specified
        if use_subset and len(dataset) > subset_size:
            indices = random.sample(range(len(dataset)), subset_size)
            self.dataset = dataset.select(indices)
        else:
            self.dataset = dataset

        print(f"Dataset loaded: {len(self.dataset)} samples")

        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = []
        self.episode_correct = []

        # Current sample
        self.current_sample = None
        self.current_label = None

        # Label mapping
        self._setup_labels()

    def _setup_labels(self) -> None:
        """Set up label mappings based on dataset."""
        # For IMDB: 0=negative, 1=positive
        if self.dataset_name == 'imdb':
            self.label_to_sentiment = {0: 'negative', 1: 'positive'}
            self.sentiment_to_label = {'negative': 0, 'positive': 1, 'neutral': -1}
            self.all_sentiments = ['negative', 'positive']
        else:
            # Generic 3-class
            self.label_to_sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
            self.sentiment_to_label = {
                'negative': 0,
                'neutral': 1,
                'positive': 2
            }
            self.all_sentiments = ['negative', 'neutral', 'positive']

    def reset(self) -> Tuple[str, int]:
        """Reset environment and get a new sample.

        Returns:
            Tuple of (text, true_label_index)
        """
        # Check if max episodes reached
        if self.max_episodes and self.current_episode >= self.max_episodes:
            raise StopIteration("Maximum episodes reached")

        # Sample random text
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]

        # Extract text and label
        if 'text' in sample:
            self.current_sample = sample['text']
        elif 'sentence' in sample:
            self.current_sample = sample['sentence']
        else:
            raise ValueError(f"Unknown text field in dataset")

        if 'label' in sample:
            self.current_label = sample['label']
        elif 'sentiment' in sample:
            self.current_label = sample['sentiment']
        else:
            raise ValueError(f"Unknown label field in dataset")

        # Truncate text if needed
        words = self.current_sample.split()[:self.max_seq_length]
        self.current_sample = ' '.join(words)

        self.current_step = 0

        return self.current_sample, self.current_label

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Predicted sentiment label

        Returns:
            Tuple of (next_text, reward, done, info)
        """
        # Convert action to label
        if isinstance(action, str):
            predicted_label = self.sentiment_to_label.get(action, -1)
        else:
            predicted_label = action

        # Compute reward
        correct = (predicted_label == self.current_label)
        reward = self.reward_correct if correct else self.reward_incorrect

        # Episode is done after one prediction (episodic task)
        done = True

        # Info dict
        info = {
            'text': self.current_sample,
            'true_label': self.current_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'true_sentiment': self.label_to_sentiment.get(
                self.current_label,
                'unknown'
            ),
            'predicted_sentiment': action
        }

        # Track statistics
        if done:
            self.episode_rewards.append(reward)
            self.episode_correct.append(int(correct))
            self.current_episode += 1

        self.current_step += 1

        # Get next sample (or same for terminal state)
        next_text = self.current_sample

        return next_text, reward, done, info

    def get_episode_stats(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get statistics for recent episodes.

        Args:
            last_n: Number of recent episodes to include (None for all)

        Returns:
            Dictionary with statistics
        """
        if not self.episode_rewards:
            return {
                'mean_reward': 0.0,
                'accuracy': 0.0,
                'total_episodes': 0
            }

        rewards = self.episode_rewards[-last_n:] if last_n else self.episode_rewards
        correct = self.episode_correct[-last_n:] if last_n else self.episode_correct

        return {
            'mean_reward': np.mean(rewards),
            'accuracy': np.mean(correct),
            'total_episodes': len(self.episode_rewards),
            'recent_episodes': len(rewards)
        }

    def get_all_sentiments(self) -> List[str]:
        """Get list of all possible sentiment labels.

        Returns:
            List of sentiment labels
        """
        return self.all_sentiments


class SimpleTokenizer:
    """Simple tokenizer for converting text to indices.

    Uses a vocabulary built from the dataset.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        max_length: int = 256,
        pad_token: str = '<PAD>',
        unk_token: str = '<UNK>'
    ):
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            pad_token: Padding token
            unk_token: Unknown token
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.word2idx: Dict[str, int] = {
            pad_token: 0,
            unk_token: 1
        }
        self.idx2word: Dict[int, str] = {
            0: pad_token,
            1: unk_token
        }
        self.vocab_built = False

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of text samples
        """
        # Count word frequencies
        word_freq: Dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.vocab_size - 2]  # Reserve 2 for special tokens

        # Build vocabulary
        for idx, (word, _) in enumerate(top_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_built = True
        print(f"Vocabulary built: {len(self.word2idx)} words")

    def encode(self, text: str) -> torch.Tensor:
        """Convert text to tensor of indices.

        Args:
            text: Input text

        Returns:
            Tensor of word indices
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")

        # Tokenize and convert to indices
        words = text.lower().split()[:self.max_length]
        indices = [
            self.word2idx.get(word, self.word2idx[self.unk_token])
            for word in words
        ]

        # Pad to max length
        if len(indices) < self.max_length:
            indices += [self.word2idx[self.pad_token]] * (self.max_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word2idx)
