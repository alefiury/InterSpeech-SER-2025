import torch
import pandas as pd

from utils.dataloader import (
    DynamicDataset,
    EmbeddingDataset,
    LastLayerEmbeddingDataset,
)


def build_dataloaders(config):
    """Builds the dataloader for the CAL-MOS model.

    Params:

    config (DictConfig): Configuration

    Returns:

    Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    train_data = pd.read_csv(config.datasets.train[0].metadata_path)
    val_data = pd.read_csv(config.datasets.val[0].metadata_path)

    # use map to build "target" column
    train_data["target"] = train_data[config.datasets.train[0].target_column].map(
        config.data.label2id
    )

    val_data["target"] = val_data[config.datasets.val[0].target_column].map(
        config.data.label2id
    )

    if config.model.model_type.lower() == "embedding":
        train_dataset = EmbeddingDataset(
            data=train_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
            use_seqaug=config.data.use_seqaug,
            data_type="train",
        )

        val_dataset = EmbeddingDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
            data_type="val",
        )
    if config.model.model_type.lower() == "last_layer_embedding":
        train_dataset = LastLayerEmbeddingDataset(
            data=train_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
        )

        val_dataset = LastLayerEmbeddingDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
        )
    else:
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

    return train_dataset, val_dataset


def get_classes_weights(config):
    train_data = pd.read_csv(config.datasets.train[0].metadata_path)
    # use map to build "target" column
    train_data["target"] = train_data[config.datasets.train[0].target_column].map(
        config.data.label2id
    )
    # Calculate class weights
    class_counts = train_data["target"].value_counts().to_dict()
    total_samples = len(train_data)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    # sort class weights by class id
    class_weights = {k: v for k, v in sorted(class_weights.items(), key=lambda item: item[0])}
    # take only the values
    class_weights = list(class_weights.values())

    class_weights = torch.tensor(class_weights).float()
    return class_weights