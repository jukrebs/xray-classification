# ViT-B/16 Head-Finetune Baseline

## Overview
- Purpose of this branch/run is to establish a meaningful AUROC baseline using a pretrained backbone before trying heavier finetunes.
- We replace the old ResNet18 with torchvision's ViT-B/16 pretrained on ImageNet (IMAGENET1K_V1 weights).
- Dataset-side preprocessing stays untouched; we adapt inside the model so we can keep all Flower wiring and dataloaders unchanged.

## Model Strategy
- Repeat the single grayscale channel to three channels inside `Net.forward`.
- Undo the dataset's `(mean=0.5, std=0.5)` normalization back to `[0, 1]`, then re-normalize with the ImageNet statistics stored in the ViT weights metadata.
- Freeze every ViT parameter (`requires_grad=False`) and replace the head with a single-logit `nn.Linear`, which becomes the only trainable component.
- Train with `BCEWithLogitsLoss`, so optimizer updates only the new head while the ViT serves as a frozen feature extractor.

## Run Configuration
- `image-size = 224` (matches ViT positional embeddings and patching).
- `lr = 1e-4` (Adam, head-only finetune).
- `local-epochs = 1`, `num-server-rounds = 15` as a safe starting point within the 15‑minute limit; increase rounds later if time budget allows.
- Use the provided `local_train.py` to sanity check per-hospital AUROC before federated runs, then launch Flower with the same configuration.

## Expected Baseline
- Should yield a materially better AUROC than random/resnet scratch thanks to ViT features while still being simple (only head learns).
- Federated logging already captures AUROC per hospital and aggregated via W&B artifacts for comparison.

## Next Steps / Ideas to Beat Baseline
1. Gradually unfreeze higher transformer blocks (e.g., last 2–4) with a lower LR while keeping earlier blocks frozen.
2. Introduce light data augmentation (histogram equalization, small rotations) compatible with the preprocessing pipeline.
3. Explore different loss weighting or threshold tuning if class imbalance is an issue.
