"""Advanced training utilities and loss functions for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal Loss for Dense Object Detection (Lin et al., 2017)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid)
            targets: Ground truth labels (0 or 1)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss with label smoothing for better generalization."""

    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing, 0.1 typical)
        """
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """Apply label smoothing and compute BCE loss."""
        # Smooth labels: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth)


class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss for handling class imbalance."""

    def __init__(self, pos_weight=None):
        """
        Args:
            pos_weight: Weight for positive class (computed from class distribution)
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """Compute weighted BCE loss."""
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight
        )


class CosineAnnealingWarmRestarts:
    """Cosine annealing learning rate schedule with warm restarts."""

    def __init__(
        self, optimizer, T_0=10, T_mult=2, eta_min=1e-6, warmup_epochs=5, base_lr=0.001
    ):
        """
        Args:
            optimizer: Optimizer instance
            T_0: Number of iterations for the first restart
            T_mult: Multiplication factor for T after each restart
            eta_min: Minimum learning rate
            warmup_epochs: Number of warmup epochs
            base_lr: Base learning rate
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            epoch_after_warmup = self.current_epoch - self.warmup_epochs
            T_cur = epoch_after_warmup % self.T_0
            lr = (
                self.eta_min
                + (self.base_lr - self.eta_min)
                * (1 + torch.cos(torch.tensor(T_cur / self.T_0 * 3.14159)))
                / 2
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_epoch += 1
        return lr


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=7, min_delta=0.0, mode="max"):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def calculate_pos_weight(dataloader):
    """Calculate positive class weight for weighted BCE loss.

    Args:
        dataloader: DataLoader for training data

    Returns:
        torch.Tensor: Weight for positive class
    """
    total_samples = 0
    positive_samples = 0

    for batch in dataloader:
        labels = batch["y"]
        total_samples += labels.numel()
        positive_samples += labels.sum().item()

    negative_samples = total_samples - positive_samples

    if positive_samples == 0:
        return torch.tensor([1.0])

    # Weight = negative_count / positive_count
    pos_weight = negative_samples / positive_samples
    return torch.tensor([pos_weight])


def get_loss_function(loss_type="bce", **kwargs):
    """Factory function to get different loss functions.

    Args:
        loss_type: One of ['bce', 'focal', 'weighted_bce', 'label_smoothing']
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Loss function instance
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    elif loss_type == "focal":
        alpha = kwargs.get("alpha", 0.25)
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == "weighted_bce":
        pos_weight = kwargs.get("pos_weight", None)
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_type == "label_smoothing":
        smoothing = kwargs.get("smoothing", 0.1)
        return LabelSmoothingBCELoss(smoothing=smoothing)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def setup_optimizer(
    model, optimizer_type="adam", lr=0.001, weight_decay=1e-4, **kwargs
):
    """Factory function to create optimizers.

    Args:
        model: Model to optimize
        optimizer_type: 'adam', 'adamw', 'sgd', 'rmsprop'
        lr: Learning rate
        weight_decay: L2 regularization
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )

    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )

    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get("nesterov", True),
        )

    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class GradientClipper:
    """Gradient clipping utility for stable training."""

    def __init__(self, max_norm=1.0, norm_type=2.0):
        """
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2 norm)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model):
        """Clip gradients of model parameters."""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        )
