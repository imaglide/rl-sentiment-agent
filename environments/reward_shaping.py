"""Reward shaping strategies for sentiment classification.

This module provides different reward shaping strategies to help
RL agents learn more effectively.
"""

from typing import Dict, Optional
import numpy as np


class RewardShaper:
    """Base class for reward shaping strategies."""

    def shape_reward(
        self,
        base_reward: float,
        info: Dict,
        **kwargs
    ) -> float:
        """Shape the reward.

        Args:
            base_reward: Base reward from environment
            info: Info dict from environment step
            **kwargs: Additional arguments

        Returns:
            Shaped reward
        """
        return base_reward


class ConfidenceRewardShaper(RewardShaper):
    """Reward shaping based on prediction confidence.

    Gives higher rewards when the agent is confident and correct,
    or penalizes more when confident but wrong.
    """

    def __init__(
        self,
        confidence_bonus: float = 0.5,
        confidence_penalty: float = 0.5
    ):
        """Initialize confidence-based reward shaper.

        Args:
            confidence_bonus: Bonus multiplier for confident correct predictions
            confidence_penalty: Penalty multiplier for confident wrong predictions
        """
        self.confidence_bonus = confidence_bonus
        self.confidence_penalty = confidence_penalty

    def shape_reward(
        self,
        base_reward: float,
        info: Dict,
        confidence: Optional[float] = None,
        **kwargs
    ) -> float:
        """Shape reward based on confidence.

        Args:
            base_reward: Base reward
            info: Info dict
            confidence: Prediction confidence (0-1)
            **kwargs: Additional arguments

        Returns:
            Shaped reward
        """
        if confidence is None:
            return base_reward

        correct = info.get('correct', False)

        if correct:
            # Bonus for confident correct predictions
            bonus = self.confidence_bonus * confidence
            return base_reward + bonus
        else:
            # Extra penalty for confident wrong predictions
            penalty = self.confidence_penalty * confidence
            return base_reward - penalty


class CurriculumRewardShaper(RewardShaper):
    """Curriculum learning: easier samples get less reward over time.

    Encourages the agent to focus on harder samples as it improves.
    """

    def __init__(
        self,
        initial_easy_bonus: float = 0.5,
        decay_rate: float = 0.01
    ):
        """Initialize curriculum reward shaper.

        Args:
            initial_easy_bonus: Initial bonus for easy samples
            decay_rate: Rate at which bonus decays
        """
        self.initial_easy_bonus = initial_easy_bonus
        self.decay_rate = decay_rate
        self.step_count = 0

    def shape_reward(
        self,
        base_reward: float,
        info: Dict,
        difficulty: Optional[float] = None,
        **kwargs
    ) -> float:
        """Shape reward based on sample difficulty.

        Args:
            base_reward: Base reward
            info: Info dict
            difficulty: Sample difficulty (0=easy, 1=hard)
            **kwargs: Additional arguments

        Returns:
            Shaped reward
        """
        self.step_count += 1

        if difficulty is None:
            return base_reward

        # Decay the easy bonus over time
        current_bonus = self.initial_easy_bonus * np.exp(-self.decay_rate * self.step_count)

        # Easy samples (low difficulty) get decreasing bonus
        if info.get('correct', False):
            easy_bonus = current_bonus * (1.0 - difficulty)
            return base_reward + easy_bonus

        return base_reward


class DenseRewardShaper(RewardShaper):
    """Provides denser rewards for partial progress.

    Instead of sparse {-1, +1}, provides intermediate rewards based
    on how "close" the prediction was.
    """

    def __init__(
        self,
        partial_reward: float = 0.3
    ):
        """Initialize dense reward shaper.

        Args:
            partial_reward: Reward for partial correctness
        """
        self.partial_reward = partial_reward

        # Define closeness between sentiments
        self.closeness = {
            ('negative', 'neutral'): 0.5,
            ('neutral', 'negative'): 0.5,
            ('neutral', 'positive'): 0.5,
            ('positive', 'neutral'): 0.5,
            ('negative', 'positive'): 0.0,  # Opposite
            ('positive', 'negative'): 0.0,
        }

    def shape_reward(
        self,
        base_reward: float,
        info: Dict,
        **kwargs
    ) -> float:
        """Shape reward to be denser.

        Args:
            base_reward: Base reward
            info: Info dict
            **kwargs: Additional arguments

        Returns:
            Shaped reward
        """
        # If already correct, return base reward
        if info.get('correct', False):
            return base_reward

        # Get predicted and true sentiments
        pred = info.get('predicted_sentiment', '')
        true = info.get('true_sentiment', '')

        # Check closeness
        key = (pred, true)
        closeness = self.closeness.get(key, 0.0)

        # Partial reward based on closeness
        if closeness > 0:
            return base_reward + (self.partial_reward * closeness)

        return base_reward


class AdaptiveRewardShaper(RewardShaper):
    """Adapts rewards based on recent performance.

    Makes rewards harder to get as performance improves.
    """

    def __init__(
        self,
        window_size: int = 100,
        min_reward: float = 0.1,
        max_reward: float = 2.0
    ):
        """Initialize adaptive reward shaper.

        Args:
            window_size: Window for computing recent accuracy
            min_reward: Minimum reward multiplier
            max_reward: Maximum reward multiplier
        """
        self.window_size = window_size
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.recent_correct = []

    def shape_reward(
        self,
        base_reward: float,
        info: Dict,
        **kwargs
    ) -> float:
        """Shape reward adaptively.

        Args:
            base_reward: Base reward
            info: Info dict
            **kwargs: Additional arguments

        Returns:
            Shaped reward
        """
        correct = info.get('correct', False)
        self.recent_correct.append(int(correct))

        # Keep only recent history
        if len(self.recent_correct) > self.window_size:
            self.recent_correct.pop(0)

        # Compute recent accuracy
        if len(self.recent_correct) < 10:
            # Not enough data, use base reward
            return base_reward

        recent_accuracy = np.mean(self.recent_correct)

        # Scale rewards inversely with accuracy
        # High accuracy -> harder to get rewards (multiplier closer to min_reward)
        # Low accuracy -> easier to get rewards (multiplier closer to max_reward)
        multiplier = self.max_reward - (self.max_reward - self.min_reward) * recent_accuracy

        return base_reward * multiplier


class MultiRewardShaper:
    """Combines multiple reward shapers.

    Applies multiple reward shaping strategies and combines them.
    """

    def __init__(
        self,
        shapers: list,
        weights: Optional[list] = None
    ):
        """Initialize multi-reward shaper.

        Args:
            shapers: List of RewardShaper instances
            weights: Optional weights for each shaper (default: equal weights)
        """
        self.shapers = shapers

        if weights is None:
            self.weights = [1.0 / len(shapers)] * len(shapers)
        else:
            assert len(weights) == len(shapers)
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def shape_reward(
        self,
        base_reward: float,
        info: Dict,
        **kwargs
    ) -> float:
        """Combine multiple reward shaping strategies.

        Args:
            base_reward: Base reward
            info: Info dict
            **kwargs: Additional arguments

        Returns:
            Combined shaped reward
        """
        shaped_rewards = []

        for shaper in self.shapers:
            shaped = shaper.shape_reward(base_reward, info, **kwargs)
            shaped_rewards.append(shaped)

        # Weighted combination
        combined = sum(
            w * r for w, r in zip(self.weights, shaped_rewards)
        )

        return combined


# Convenience function to create shapers
def create_reward_shaper(strategy: str = 'none', **kwargs) -> RewardShaper:
    """Create a reward shaper by strategy name.

    Args:
        strategy: Strategy name ('none', 'confidence', 'curriculum', 'dense', 'adaptive')
        **kwargs: Arguments for the shaper

    Returns:
        RewardShaper instance
    """
    if strategy == 'none':
        return RewardShaper()
    elif strategy == 'confidence':
        return ConfidenceRewardShaper(**kwargs)
    elif strategy == 'curriculum':
        return CurriculumRewardShaper(**kwargs)
    elif strategy == 'dense':
        return DenseRewardShaper(**kwargs)
    elif strategy == 'adaptive':
        return AdaptiveRewardShaper(**kwargs)
    else:
        raise ValueError(f"Unknown reward shaping strategy: {strategy}")
