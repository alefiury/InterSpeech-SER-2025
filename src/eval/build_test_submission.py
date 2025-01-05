import os
import sys

# Adds the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pprint import pprint

import torch
import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from models.pl_wrapper import PLWrapper
from utils.dataloader import (
    DynamicDataset,
    DynamicAudioTextDataset,
    EmbeddingDataset,
    LastLayerEmbeddingDataset,
)

from utils.dataloader import (
    DynamicCollate,
    DynamicAudioTextCollate,
    XEUSNestCollate,
    EmbeddingCollate,
    LastLayerEmbeddingCollate
)


def build_dataloaders(
    config: DictConfig,
    test_data: pd.DataFrame,
    filename_column: str,
    target_column: str,
    base_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Builds the dataloader for the CAL-MOS model.

    Params:

    config (DictConfig): Configuration

    Returns:

    Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    # use map to build "target" column
    test_data["target"] = test_data[target_column].map(
        config.data.label2id
    )

    if config.model.model_type.lower() == "embedding":
        test_dataset = EmbeddingDataset(
            data=test_data,
            filename_column=filename_column,
            target_column="target",
            base_dir=base_dir,
            data_type="test",
        )
        collate_fn = EmbeddingCollate()
    if config.model.model_type.lower() == "last_layer_embedding":
        test_dataset = LastLayerEmbeddingDataset(
            data=test_data,
            filename_column=filename_column,
            target_column="target",
            base_dir=base_dir,
        )
        collate_fn = LastLayerEmbeddingCollate()
    elif config.model.model_type.lower() == "dynamic" \
            or config.model.model_type.lower() == "nest" \
            or config.model.model_type.lower() == "xeus":
        test_dataset = DynamicDataset(
            data=test_data,
            filename_column=filename_column,
            target_column="target",
            base_dir=base_dir,
            mixup_alpha=config.data.mixup_alpha,
            data_type="test",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )
        if config.model.model_type.lower() == "xeus" or config.model.model_type.lower() == "nest":
            collate_fn = XEUSNestCollate()
        elif config.model.model_type.lower() == "dynamic":
            processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
            collate_fn = DynamicCollate(
                target_sr=config.data.target_sr,
                processor=processor,
            )
    elif config.model.model_type.lower() == "dynamic_audio_text":
        test_dataset = DynamicAudioTextDataset(
            data=test_data,
            filename_column=config.datasets.train[0].filename_column,
            transcript_column=config.datasets.train[0].transcript_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
            mixup_alpha=config.data.mixup_alpha,
            data_type="test",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )
        processor = AutoFeatureExtractor.from_pretrained(config.model.audio_model_name)
        text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
        collate_fn = DynamicAudioTextCollate(
            target_sr=config.data.target_sr,
            processor=processor,
            text_tokenizer=text_tokenizer,
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return test_dataloader

@torch.no_grad()
def evaluate_model(
    config: DictConfig,
    test_dataloader: DataLoader,
    checkpoint_path: str,
    device: str = "cuda:0"
):
    """Evaluates the model on the test set.

    Params:

    config (DictConfig): Configuration

    test_dataloader (DataLoader): Test dataloader

    checkpoint_path (str): Path to the checkpoint

    gpu (int): GPU number

    Returns:

    None
    """
    model = PLWrapper.load_from_checkpoint(checkpoint_path, config=config, map_location=device)
    model = model.to(device)
    model.eval()

    # targets = []
    predictions = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        inputs, target = batch

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = [i.to(device) for i in inputs]
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model(inputs)
        # apply softmax and argmax
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)

        # targets.extend(target.cpu().numpy())
        predictions.extend(output.cpu().numpy())

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test-metadata-path",
        required=True,
        type=str,
        help="Path to the test metadata file (csv)"
    )
    parser.add_argument(
        "--filename-column",
        default="filename",
        type=str,
        help="Name of the column with the filenames"
    )
    parser.add_argument(
        "--target-column",
        default="emotion",
        type=str,
        help="Name of the column with the target values"
    )
    parser.add_argument(
        "--base-dir",
        default="../data/ser-2025",
        type=str,
        help="Base directory for the audio files"
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default=0,
        type=int
    )
    parser.add_argument(
        "-ck",
        "--checkpoint-path",
        required=False,
        type=str,
        default="../checkpoints/ser-2025",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--num-workers",
        default=12,
        type=int,
        help="Number of workers"
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    test_data = pd.read_csv(args.test_metadata_path)
    test_dataloader = build_dataloaders(
        config=config,
        test_data=test_data,
        filename_column=args.filename_column,
        target_column=args.target_column,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    predictions = evaluate_model(
        config=config,
        test_dataloader=test_dataloader,
        checkpoint_path=args.checkpoint_path,
        device=f"cuda:{args.gpu}"
    )

    df_test_metadata = pd.read_csv(args.test_metadata_path)

    filenames = df_test_metadata[args.filename_column].tolist()

    label2id = config.data.label2id

    id2label = {v: k for k, v in label2id.items()}

    print(id2label)

    predictions_labels = [id2label[p] for p in predictions]

    df_submission = pd.DataFrame(
        {
            "FileName": filenames,
            "EmoClass": predictions_labels
        }
    )

    df_submission.to_csv("/hadatasets/alef.ferreira/SER/Interspeech/submission-bimodal-3dit02ei.csv", index=False)



if __name__ == "__main__":
    main()
