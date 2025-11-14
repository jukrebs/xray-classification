"""Data augmentation strategies for medical imaging."""

import torch
from torchvision import transforms


class MedicalImageAugmentation:
    """Medical imaging specific augmentation pipeline."""

    def __init__(
        self,
        image_size=128,
        rotation_degrees=10,
        brightness=0.2,
        contrast=0.2,
        enable_mixup=False,
        mixup_alpha=0.2,
    ):
        """
        Args:
            image_size: Target image size
            rotation_degrees: Range for random rotation
            brightness: Brightness adjustment factor
            contrast: Contrast adjustment factor
            enable_mixup: Whether to apply mixup augmentation
            mixup_alpha: Alpha parameter for mixup (beta distribution)
        """
        self.image_size = image_size
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha

        # Training augmentation pipeline
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                # Random horizontal flip (anatomically valid for chest X-rays)
                transforms.RandomHorizontalFlip(p=0.5),
                # Small random rotation
                transforms.RandomRotation(degrees=rotation_degrees),
                # Random affine transformations
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # Small translations
                    scale=(0.95, 1.05),  # Small scaling
                ),
                # Brightness and contrast
                transforms.ColorJitter(brightness=brightness, contrast=contrast),
                # Random crop with padding
                transforms.RandomCrop(image_size, padding=8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        # Validation/test transform (no augmentation)
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __call__(self, batch, is_training=True):
        """Apply augmentation to a batch.

        Args:
            batch: Batch from dataset with 'image' and 'label' fields
            is_training: Whether in training mode

        Returns:
            Dict with 'x' (images) and 'y' (labels)
        """
        transform = self.train_transform if is_training else self.eval_transform

        # Transform images
        transformed_images = [transform(img) for img in batch["image"]]
        result = {"x": torch.stack(transformed_images)}

        # Process labels
        labels = []
        for label_list in batch["label"]:
            has_finding = not (len(label_list) == 1 and label_list[0] == "No Finding")
            labels.append(torch.tensor([float(has_finding)]))
        result["y"] = torch.stack(labels)

        # Apply mixup if enabled and training
        if self.enable_mixup and is_training:
            result = self.mixup_batch(result)

        return result

    def mixup_batch(self, batch):
        """Apply mixup augmentation to a batch.

        Mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
        """
        x, y = batch["x"], batch["y"]
        batch_size = x.size(0)

        if batch_size < 2:
            return batch

        # Sample lambda from beta distribution
        if self.mixup_alpha > 0:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size)

        # Mix inputs and labels
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        return {"x": mixed_x, "y": mixed_y}


class CutMixAugmentation:
    """CutMix augmentation for medical images.

    CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    """

    def __init__(self, alpha=1.0, prob=0.5):
        """
        Args:
            alpha: CutMix alpha parameter
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        """Apply CutMix to a batch."""
        if torch.rand(1) > self.prob:
            return batch

        x, y = batch["x"], batch["y"]
        batch_size = x.size(0)

        if batch_size < 2:
            return batch

        # Sample lambda
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()

        # Random permutation
        index = torch.randperm(batch_size)

        # Get random box
        _, _, h, w = x.size()
        cut_rat = torch.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Random center point
        cx = torch.randint(0, w, (1,)).item()
        cy = torch.randint(0, h, (1,)).item()

        # Get box coordinates
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)

        # Apply CutMix
        x_cutmix = x.clone()
        x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        y_cutmix = lam * y + (1 - lam) * y[index]

        return {"x": x_cutmix, "y": y_cutmix}


class RandomErasing:
    """Random erasing augmentation to improve robustness."""

    def __init__(self, prob=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)):
        """
        Args:
            prob: Probability of applying random erasing
            scale: Range of proportion of erased area
            ratio: Range of aspect ratio of erased area
        """
        self.transform = transforms.RandomErasing(
            p=prob, scale=scale, ratio=ratio, value="random"
        )

    def __call__(self, x):
        """Apply random erasing to tensor."""
        return self.transform(x)


def get_augmentation_pipeline(
    image_size=128,
    augmentation_type="standard",
    rotation_degrees=10,
    enable_mixup=False,
    enable_cutmix=False,
    enable_erasing=False,
):
    """Factory function to create augmentation pipelines.

    Args:
        image_size: Target image size
        augmentation_type: 'none', 'light', 'standard', 'heavy'
        rotation_degrees: Rotation range
        enable_mixup: Enable mixup augmentation
        enable_cutmix: Enable cutmix augmentation
        enable_erasing: Enable random erasing

    Returns:
        Augmentation instance or transform
    """
    if augmentation_type == "none":
        return None

    # Adjust parameters based on augmentation intensity
    aug_params = {
        "light": {"rotation": 5, "brightness": 0.1, "contrast": 0.1},
        "standard": {"rotation": 10, "brightness": 0.2, "contrast": 0.2},
        "heavy": {"rotation": 15, "brightness": 0.3, "contrast": 0.3},
    }

    params = aug_params.get(augmentation_type, aug_params["standard"])

    augmenter = MedicalImageAugmentation(
        image_size=image_size,
        rotation_degrees=params["rotation"],
        brightness=params["brightness"],
        contrast=params["contrast"],
        enable_mixup=enable_mixup,
    )

    # Add additional augmentations
    if enable_cutmix:
        cutmix = CutMixAugmentation(alpha=1.0, prob=0.5)
        # Wrap both augmenters
        original_call = augmenter.__call__

        def combined_call(batch, is_training=True):
            batch = original_call(batch, is_training)
            if is_training:
                batch = cutmix(batch)
            return batch

        augmenter.__call__ = combined_call

    if enable_erasing:
        erasing = RandomErasing(prob=0.5)
        # Apply after other augmentations
        original_call = augmenter.__call__

        def with_erasing(batch, is_training=True):
            batch = original_call(batch, is_training)
            if is_training:
                batch["x"] = torch.stack([erasing(img) for img in batch["x"]])
            return batch

        augmenter.__call__ = with_erasing

    return augmenter
