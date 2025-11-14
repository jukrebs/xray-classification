"""Local training script for testing the X-ray classification pipeline."""

import torch
from berlin25_xray.task import Net, load_data
from berlin25_xray.task import test as test_fn
from berlin25_xray.task import train as train_fn

HOSPITAL = "A"  # A, B, or C
EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 16
MAX_BATCHES = None  # For debugging, set to None to use all batches


def main():
    """Train X-ray classifier locally on a specific hospital's data."""
    device = torch.device("cpu")

    hospital_map = {"A": 0, "B": 1, "C": 2}
    partition_id = hospital_map[HOSPITAL]

    print(f"\nLoading Hospital{HOSPITAL} data...")
    trainloader, evalloader = load_data(
        partition_id,
        batch_size=BATCH_SIZE,
        max_batches=MAX_BATCHES,
        preprocessed=True,
    )

    print(f"Train batches: {len(trainloader)}, Eval batches: {len(evalloader)}")
    if MAX_BATCHES:
        print(f"Limited to {MAX_BATCHES} batches per epoch")

    model = Net()
    model.to(device)
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        # Train for 1 epoch
        print(f"\n[Epoch {epoch}/{EPOCHS}] Training...")
        train_loss = train_fn(
            model, trainloader, epochs=1, lr=LEARNING_RATE, device=device
        )
        print(f"  Train loss: {train_loss:.4f}")

        # Evaluate after each epoch
        print(f"[Epoch {epoch}/{EPOCHS}] Evaluating...")
        eval_loss, auroc, f1, sensitivity, specificity, precision = test_fn(
            model, evalloader, device
        )
        print(f"  Eval loss: {eval_loss:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")

    # Save model
    model_path = f"hospital_{HOSPITAL}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete! Model saved to {model_path}")


if __name__ == "__main__":
    main()
