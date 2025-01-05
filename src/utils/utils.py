import torch
import pandas as pd

from utils.dataloader import (
    DynamicDataset,
    DynamicAudioTextDataset,
    DynamicAudioTextSpeakerEmbDataset,
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
    elif config.model.model_type.lower() == "dynamic" \
            or config.model.model_type.lower() == "nest" \
            or config.model.model_type.lower() == "xeus":
        train_dataset = DynamicDataset(
            data=train_data,
            base_dir=config.datasets.train[0].base_dir,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            mixup_alpha=config.data.mixup_alpha,
            use_rand_truncation=config.data.use_rand_truncation,
            min_duration=config.data.min_duration,
            data_type="train",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
            # background noise parameters
            use_background_noise=config.data.use_background_noise,
            background_noise_dir=config.data.background_noise_dir,
            background_noise_min_snr_in_db=config.data.background_noise_min_snr_in_db,
            background_noise_max_snr_in_db=config.data.background_noise_max_snr_in_db,
            background_noise_p=config.data.background_noise_p,
            # impulse response parameters
            use_rir=config.data.use_rir,
            rir_dir=config.data.rir_dir,
            rir_p=config.data.rir_p,
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
    elif config.model.model_type.lower() == "dynamic_audio_text":
        train_dataset = DynamicAudioTextDataset(
            data=train_data,
            base_dir=config.datasets.train[0].base_dir,
            filename_column=config.datasets.train[0].filename_column,
            transcript_column=config.datasets.train[0].transcript_column,
            target_column="target",
            mixup_alpha=config.data.mixup_alpha,
            use_rand_truncation=config.data.use_rand_truncation,
            min_duration=config.data.min_duration,
            data_type="train",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
            # background noise parameters
            use_background_noise=config.data.use_background_noise,
            background_noise_dir=config.data.background_noise_dir,
            background_noise_min_snr_in_db=config.data.background_noise_min_snr_in_db,
            background_noise_max_snr_in_db=config.data.background_noise_max_snr_in_db,
            background_noise_p=config.data.background_noise_p,
            # impulse response parameters
            use_rir=config.data.use_rir,
            rir_dir=config.data.rir_dir,
            rir_p=config.data.rir_p,
            # text augmentation parameters
            use_text_augmentation=config.data.use_text_augmentation,
            text_augmentation_p=config.data.text_augmentation_p,
        )

        val_dataset = DynamicAudioTextDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            transcript_column=config.datasets.train[0].transcript_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
            mixup_alpha=config.data.mixup_alpha,
            data_type="val",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )
    elif config.model.model_type.lower() == "dynamic_audio_text_speakeremb" or config.model.model_type.lower() == "dynamic_audio_text_speakeremb_melspec":
        train_dataset = DynamicAudioTextSpeakerEmbDataset(
            data=train_data,
            base_dir=config.datasets.train[0].base_dir,
            filename_column=config.datasets.train[0].filename_column,
            transcript_column=config.datasets.train[0].transcript_column,
            speakeremb_base_dir=config.datasets.train[0].speakeremb_base_dir,
            target_column="target",
            mixup_alpha=config.data.mixup_alpha,
            use_rand_truncation=config.data.use_rand_truncation,
            min_duration=config.data.min_duration,
            data_type="train",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
            # background noise parameters
            use_background_noise=config.data.use_background_noise,
            background_noise_dir=config.data.background_noise_dir,
            background_noise_min_snr_in_db=config.data.background_noise_min_snr_in_db,
            background_noise_max_snr_in_db=config.data.background_noise_max_snr_in_db,
            background_noise_p=config.data.background_noise_p,
            # impulse response parameters
            use_rir=config.data.use_rir,
            rir_dir=config.data.rir_dir,
            rir_p=config.data.rir_p,
        )

        val_dataset = DynamicAudioTextSpeakerEmbDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            transcript_column=config.datasets.train[0].transcript_column,
            speakeremb_base_dir=config.datasets.train[0].speakeremb_base_dir,
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