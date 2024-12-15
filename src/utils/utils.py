import torch
import pandas as pd
from transformers import AutoFeatureExtractor

from utils.dataloader import DynamicDataset, DynamicCollate, XEUSNestCollate


def build_dataloaders(config):
    """Builds the dataloader for the CAL-MOS model.

    Params:

    config (DictConfig): Configuration

    Returns:

    Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    train_data = pd.read_csv(config.datasets.train[0].metadata_path)
    val_data = pd.read_csv(config.datasets.val[0].metadata_path)

    if config.model.model_type.lower() == "xeus" or config.model.model_type.lower() == "nest":
        collate_fn = XEUSNestCollate()
    else:
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate_fn = DynamicCollate(
            target_sr=config.data.target_sr,
            processor=processor,
        )

    # use map to build "target" column
    train_data["target"] = train_data[config.datasets.train[0].target_column].map(
        config.data.label2id
    )

    val_data["target"] = val_data[config.datasets.val[0].target_column].map(
        config.data.label2id
    )

    train_dataset = DynamicDataset(
        data=train_data,
        filename_column=config.datasets.train[0].filename_column,
        target_column="target",
        base_dir=config.datasets.train[0].base_dir,
        mixup_alpha=config.data.mixup_alpha,
        use_rand_truncation=config.data.use_rand_truncation,
        min_duration=config.data.min_duration,
        insert_white_noise=config.data.insert_white_noise,
        min_white_noise_amp=config.data.min_white_noise_amp,
        max_white_noise_amp=config.data.max_white_noise_amp,
        data_type="train",
        class_num=config.data.num_classes,
        target_sr=config.data.target_sr,
    )

    val_dataset = DynamicDataset(
        data=val_data,
        filename_column=config.datasets.train[0].filename_column,
        target_column="target",
        base_dir=config.datasets.train[0].base_dir,
        mixup_alpha=config.data.mixup_alpha,
        data_type="val",
        class_num=config.data.num_classes,
        target_sr=config.data.target_sr,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader