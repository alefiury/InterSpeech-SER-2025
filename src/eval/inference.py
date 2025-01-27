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
from transformers import AutoFeatureExtractor, AutoTokenizer, BatchEncoding
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from models.pl_wrapper import PLWrapper
from utils.dataloader import (
    DynamicDataset,
    DynamicAudioTextDataset,
    DynamicAudioTextSpeakerEmbDataset,
    EmbeddingDataset,
    LastLayerEmbeddingDataset,
    LastLayerEmbeddingTextDataset,
    LastLayerEmbeddingTextMelSpecDataset,
    BimodalEmbeddingDataset,
    BimodalEmbeddingMelSpecDataset,
    BimodalEmbeddingF0Dataset,
    BimodalEmbeddingF0MelSpecDataset,
)

from utils.dataloader import (
    DynamicCollate,
    DynamicAudioTextCollate,
    DynamicAudioTextSpeakerEmbCollate,
    XEUSNestCollate,
    EmbeddingCollate,
    LastLayerEmbeddingCollate,
    XEUSNestTextCollate,
    XEUSNestTextSpeakerEmbCollate,
    LastLayerEmbeddingTextCollate,
    LastLayerEmbeddingTextMelSpecCollate,
    BimodalEmbeddingCollate,
    BimodalEmbeddingMelSpecCollate,
    BimodalEmbeddingF0Collate,
    BimodalEmbeddingF0MelSpecCollate,
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

    # check if "gender2id" exists in the config.data
    if "gender2id" in config.data:
        test_data["gender_id"] = test_data[config.datasets.val[0].gender_column].map(
            config.data.gender2id
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
    elif config.model.model_type.lower() == "dynamic_audio_text" \
        or config.model.model_type.lower() == "xeus_text":
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
        text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
        if config.model.model_type.lower() == "dynamic_audio_text":
            processor = AutoFeatureExtractor.from_pretrained(config.model.audio_model_name)
            collate_fn = DynamicAudioTextCollate(
                target_sr=config.data.target_sr,
                processor=processor,
                text_tokenizer=text_tokenizer,
            )
        elif config.model.model_type.lower() == "xeus_text":
            collate_fn = XEUSNestTextCollate(
                text_tokenizer=text_tokenizer,
            )
    elif config.model.model_type.lower() == "dynamic_audio_text_speakeremb" \
        or config.model.model_type.lower() == "dynamic_audio_text_speakeremb_melspec" \
            or config.model.model_type.lower() == "xeus_text_speakeremb" \
                or config.model.model_type.lower() == "xeus_text_speakeremb_melspec":
        test_dataset = DynamicAudioTextSpeakerEmbDataset(
            data=test_data,
            filename_column=config.datasets.train[0].filename_column,
            transcript_column=config.datasets.train[0].transcript_column,
            gender_column="gender_id",
            speakeremb_base_dir=config.datasets.train[0].speakeremb_base_dir,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
            mixup_alpha=config.data.mixup_alpha,
            data_type="test",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )
        text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)

        if config.model.model_type.lower() == "dynamic_audio_text_speakeremb" \
            or config.model.model_type.lower() == "dynamic_audio_text_speakeremb_melspec":
                processor = AutoFeatureExtractor.from_pretrained(config.model.audio_model_name)
                collate_fn = DynamicAudioTextSpeakerEmbCollate(
                    target_sr=config.data.target_sr,
                    processor=processor,
                    text_tokenizer=text_tokenizer,
                )
        elif config.model.model_type.lower() == "xeus_text_speakeremb" \
            or config.model.model_type.lower() == "xeus_text_speakeremb_melspec":
                collate_fn = XEUSNestTextSpeakerEmbCollate(
                    text_tokenizer=text_tokenizer,
                )

    elif config.model.model_type.lower() == "last_layer_embedding_text":
        test_dataset = LastLayerEmbeddingTextDataset(
            data=test_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            transcript_column=config.datasets.train[0].transcript_column,
            gender_column="gender_id",
            base_dir=config.datasets.train[0].base_dir,
            data_type="test",
        )

        text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
        collate_fn = LastLayerEmbeddingTextCollate(text_tokenizer=text_tokenizer)
    elif config.model.model_type.lower() == "last_layer_embedding_text_melspec":
        test_dataset = LastLayerEmbeddingTextMelSpecDataset(
            data=test_data,
            target_sr=config.data.target_sr,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            transcript_column=config.datasets.train[0].transcript_column,
            gender_column="gender_id",
            audio_base_dir=config.datasets.train[0].audio_base_dir,
            base_dir=config.datasets.train[0].base_dir,
            data_type="test",
        )

        text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
        collate_fn = LastLayerEmbeddingTextMelSpecCollate(text_tokenizer=text_tokenizer)

    elif config.model.model_type.lower() == "bimodal_embedding":
        test_dataset = BimodalEmbeddingDataset(
            data=test_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            audio_base_dir=config.datasets.train[0].audio_base_dir,
            text_base_dir=config.datasets.train[0].text_base_dir,
            data_type="test",
        )

        collate_fn = BimodalEmbeddingCollate()
    elif config.model.model_type.lower() == "bimodal_embedding_f0_melspec":
        test_dataset = BimodalEmbeddingF0MelSpecDataset(
            data=test_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column="target",
            base_dir=config.datasets.train[0].base_dir,
            audio_base_dir=config.datasets.train[0].audio_base_dir,
            text_base_dir=config.datasets.train[0].text_base_dir,
            f0_base_dir=config.datasets.train[0].f0_base_dir,
            target_sr=config.data.target_sr,
            data_type="test",
        )
        collate_fn = BimodalEmbeddingF0MelSpecCollate()

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

    targets = []
    predictions = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        inputs, target = batch

        # print(inputs)

        if isinstance(inputs, torch.Tensor):
            # If inputs is a single tensor
            inputs = inputs.to(device)
        elif isinstance(inputs, (list, tuple)):
            # If inputs is a list or tuple, process each element based on its type
            processed_inputs = []
            for i in inputs:
                if isinstance(i, dict):
                    # If the element is a dictionary, move each tensor to the device
                    processed_dict = {k: v.to(device) for k, v in i.items()}
                    processed_inputs.append(processed_dict)
                else:
                    # If the element is a tensor, move it to the device
                    processed_inputs.append(i.to(device))
                # else:
                #     raise TypeError(f"Unsupported type in inputs tuple: {type(i)}")
            inputs = tuple(processed_inputs)
        elif isinstance(inputs, dict):
            # If inputs is a dictionary, move each tensor to the device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            raise TypeError(f"Unsupported type for inputs: {type(inputs)}")

        output = model(inputs)
        # apply softmax and argmax
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)

        targets.extend(target.cpu().numpy())
        predictions.extend(output.cpu().numpy())

    return targets, predictions


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

    targets, predictions = evaluate_model(
        config=config,
        test_dataloader=test_dataloader,
        checkpoint_path=args.checkpoint_path,
        device=f"cuda:{args.gpu}"
    )

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"results-{args.checkpoint_path.split('/')[-4]}-{os.path.basename(args.test_metadata_path)[:-4]}"
    output_filepath = os.path.join(output_dir, filename)

    df_test_metadata = pd.read_csv(args.test_metadata_path)

    filenames = df_test_metadata[args.filename_column].tolist()

    df_submission = pd.DataFrame(
        {
            "FileName": filenames,
            "pred": predictions,
            "target": targets,
        }
    )

    df_submission.to_csv(output_filepath+".csv", index=False)

    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro")
    f1_micro = f1_score(targets, predictions, average="micro")
    recall = recall_score(targets, predictions, average="macro")
    precision = precision_score(targets, predictions, average="macro")

    print(f"Accuracy: {accuracy}")
    print(f"F1-Macro: {f1}")
    print(f"F1-Micro: {f1_micro}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

    # write file with results
    with open(output_filepath+".txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1-Macro: {f1}\n")
        f.write(f"F1-Micro: {f1_micro}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}")

    # plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    label2id = config.data.label2id
    id2label = {v: k for k, v in label2id.items()}

    labels = [id2label[i] for i in range(len(id2label))]

    cm = confusion_matrix(targets, predictions)
    # put labels on x and y axis
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_filepath+"-confusion_matrix.png")

if __name__ == "__main__":
    main()
