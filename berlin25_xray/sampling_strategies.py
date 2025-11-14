"""Advanced sampling strategies for federated learning."""

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class BalancedBatchSampler(Sampler):
    """Sampler that creates balanced batches with equal class representation.

    Useful for handling class imbalance in medical imaging datasets.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: Dataset with 'y' field containing labels
            batch_size: Size of each batch (should be even for binary classification)
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Get all labels
        labels = []
        for i in range(len(dataset)):
            label = dataset[i]["y"]
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, (list, np.ndarray)):
                label = label[0]
            labels.append(int(label))

        # Separate indices by class
        self.positive_indices = [i for i, label in enumerate(labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(labels) if label == 0]

        # Calculate number of batches
        min_class_size = min(len(self.positive_indices), len(self.negative_indices))
        self.num_samples = (min_class_size * 2 // batch_size) * batch_size

    def __iter__(self):
        # Shuffle indices
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        # Create balanced batches
        samples_per_class = self.batch_size // 2
        batches = []

        for i in range(0, self.num_samples // 2, samples_per_class):
            batch = []
            # Add positive samples
            batch.extend(self.positive_indices[i : i + samples_per_class])
            # Add negative samples
            batch.extend(self.negative_indices[i : i + samples_per_class])
            # Shuffle within batch
            np.random.shuffle(batch)
            batches.extend(batch)

        return iter(batches)

    def __len__(self):
        return self.num_samples


class ImportanceSampler(Sampler):
    """Sampler that focuses on hard/important examples based on loss values.

    This requires tracking per-sample losses during training.
    """

    def __init__(
        self, dataset, loss_history, batch_size, temperature=2.0, min_prob=0.01
    ):
        """
        Args:
            dataset: Dataset
            loss_history: Dict mapping sample index to recent loss values
            batch_size: Batch size
            temperature: Higher = more uniform, lower = more focused on hard samples
            min_prob: Minimum sampling probability for any sample
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

        # Calculate sampling probabilities based on losses
        losses = np.array([loss_history.get(i, 1.0) for i in range(self.num_samples)])

        # Apply temperature scaling
        losses = losses ** (1.0 / temperature)

        # Normalize to probabilities with minimum threshold
        probs = losses / losses.sum()
        probs = np.maximum(probs, min_prob)
        probs = probs / probs.sum()

        self.weights = torch.from_numpy(probs).float()

    def __iter__(self):
        # Sample with replacement based on importance weights
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_class_balanced_sampler(dataset, batch_size=None):
    """Create a weighted sampler for class-balanced sampling.

    Args:
        dataset: Dataset with 'y' field
        batch_size: Optional batch size for BalancedBatchSampler

    Returns:
        WeightedRandomSampler or BalancedBatchSampler
    """
    # Extract labels
    labels = []
    for i in range(len(dataset)):
        label = dataset[i]["y"]
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, (list, np.ndarray)):
            label = label[0]
        labels.append(int(label))

    labels = np.array(labels)

    # Calculate class weights (inverse frequency)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    if batch_size is not None:
        # Use balanced batch sampler
        return BalancedBatchSampler(dataset, batch_size)
    else:
        # Use weighted random sampler
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )


class FederatedAwareSampler(Sampler):
    """Sampler that accounts for data heterogeneity in federated learning.

    Oversamples minority classes and undersamples majority classes to
    reduce the effect of non-IID data distribution.
    """

    def __init__(self, dataset, target_distribution=None, num_samples=None):
        """
        Args:
            dataset: Dataset
            target_distribution: Desired class distribution (dict: class -> proportion)
            num_samples: Total number of samples to generate per epoch
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)

        # Get current distribution
        labels = []
        for i in range(len(dataset)):
            label = dataset[i]["y"]
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, (list, np.ndarray)):
                label = label[0]
            labels.append(int(label))

        labels = np.array(labels)
        current_dist = np.bincount(labels) / len(labels)

        # Use uniform distribution if not specified
        if target_distribution is None:
            target_distribution = {
                i: 1.0 / len(current_dist) for i in range(len(current_dist))
            }

        # Calculate sampling weights to match target distribution
        weights = np.zeros(len(dataset))
        for class_idx in range(len(current_dist)):
            class_mask = labels == class_idx
            target_prop = target_distribution.get(class_idx, current_dist[class_idx])
            current_count = class_mask.sum()
            if current_count > 0:
                weights[class_mask] = target_prop / current_count

        # Normalize
        self.weights = torch.from_numpy(weights).float()
        self.weights = self.weights / self.weights.sum() * self.num_samples

    def __iter__(self):
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=True
        ).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


def create_sampler(
    dataset, strategy="balanced", batch_size=None, loss_history=None, **kwargs
):
    """Factory function to create different sampling strategies.

    Args:
        dataset: Dataset to sample from
        strategy: One of ['balanced', 'importance', 'federated', 'weighted', None]
        batch_size: Batch size for batch-based samplers
        loss_history: Loss history for importance sampling
        **kwargs: Additional arguments for specific samplers

    Returns:
        Sampler instance or None
    """
    if strategy is None or strategy == "none":
        return None

    if strategy == "balanced":
        if batch_size:
            return BalancedBatchSampler(dataset, batch_size)
        else:
            return get_class_balanced_sampler(dataset)

    elif strategy == "importance":
        if loss_history is None:
            raise ValueError("loss_history required for importance sampling")
        return ImportanceSampler(
            dataset,
            loss_history,
            batch_size or len(dataset),
            temperature=kwargs.get("temperature", 2.0),
        )

    elif strategy == "federated":
        return FederatedAwareSampler(
            dataset,
            target_distribution=kwargs.get("target_distribution"),
            num_samples=kwargs.get("num_samples"),
        )

    elif strategy == "weighted":
        return get_class_balanced_sampler(dataset, batch_size=None)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
