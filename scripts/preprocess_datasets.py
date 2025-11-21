"""Preprocess X-ray datasets by applying transforms once and saving as tensors.

We used this script to produce the preprocessed versions of the X-Ray dataset.
This speeds up training by avoiding repeated on-the-fly transformations.

The X-Ray dataset is based on https://huggingface.co/datasets/BahaaEldin0/NIH-Chest-Xray-14,
from which we artificially created non-iid distributions across the different hospitals.

Usage:
    python preprocess_datasets.py --image-size 224
    python preprocess_datasets.py --image-size 128
"""

import argparse
import os
from glob import glob

import torch
from datasets import DatasetDict, load_from_disk
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xray/raw"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xray"))


def apply_transforms(batch, image_size):
    """For reference: This is the apply_transforms we used for image preprocessing."""
    result = {}

    _transform_pipeline = Compose(
        [
            Resize((image_size, image_size)),
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale
        ]
    )

    # Transform images and stack them into a tensor
    transformed_images = [_transform_pipeline(img) for img in batch["image"]]
    result["x"] = torch.stack(transformed_images)

    # Binary classification: 0 for "No Finding", 1 for any finding
    labels = []
    for label_list in batch["label"]:
        # If "No Finding" is the only label, it's 0; otherwise it's 1
        has_finding = not (len(label_list) == 1 and label_list[0] == "No Finding")
        labels.append(torch.tensor([float(has_finding)]))
    result["y"] = torch.stack(labels)

    return result


def transform_split(split_dataset, split_name, image_size):
    """Utility: Transform a single split (train/eval)."""
    transformed = split_dataset.map(
        lambda batch: apply_transforms(batch, image_size),
        batched=True,
        batch_size=256,
        num_proc=12,
        desc=split_name,
    )
    return transformed.remove_columns(["image", "label"])


def preprocess_dataset(dataset_name, image_size):
    """Preprocess a single dataset."""
    dataset_path = f"{SRC_DIR}/{dataset_name}"
    output_path = f"{OUTPUT_DIR}/preprocessed_{image_size}/{dataset_name}"

    if os.path.exists(output_path):
        print(f"\n# {dataset_name}: Already preprocessed, skipping")
        return

    print(
        f"\n# Processing {dataset_name} (image_size={image_size}) and saving to {output_path}"
    )
    dataset = load_from_disk(dataset_path)
    preprocessed = DatasetDict(
        {
            split_name: transform_split(ds, split_name, image_size)
            for split_name, ds in dataset.items()
        }
    )
    preprocessed.save_to_disk(output_path)


def main():
    """Main function to preprocess all datasets."""
    parser = argparse.ArgumentParser(description="Preprocess X-ray datasets")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    dataset_paths = glob(f"{SRC_DIR}/Hospital*") + glob(f"{SRC_DIR}/Test")
    dataset_names = [os.path.basename(path) for path in dataset_paths]

    for dataset_name in dataset_names:
        preprocess_dataset(dataset_name, args.image_size)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
