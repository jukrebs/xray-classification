import os

from datasets import load_from_disk
from torch.utils.data import DataLoader

from berlin25_xray.task import collate_preprocessed


def get_partition_loaders(part_id: int, image_size: int = 128, batch_size: int = 16):
    PARTITION_HOSPITAL_MAP = {
        0: "A",
        1: "B",
        2: "C",
    }
    
    dataset_name = f"Hospital{PARTITION_HOSPITAL_MAP[part_id]}"
    dataset_dir = os.environ["DATASET_DIR"]
    dataset_path = (
        f"{dataset_dir}/xray_fl_datasets_preprocessed_{image_size}/{dataset_name}"
    )
    
    full_dataset = load_from_disk(dataset_path)
    train_data = full_dataset["train"]
    val_data = full_dataset["eval"]
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_preprocessed,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_preprocessed,
        persistent_workers=True,
    )
    
    return train_loader, val_loader
