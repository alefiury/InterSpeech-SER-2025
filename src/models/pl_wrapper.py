import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import DictConfig
import numpy as np
from torch.optim import Adam, AdamW
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer, AutoFeatureExtractor
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection, ConfusionMatrix

# backward compatibility
try:
    from kornia.losses import FocalLoss
except ImportError:
    FocalLoss = None

from models.factory import create_ser_model
from utils.schedulers import CosineWarmupLR, LinearLR
from utils.utils import build_dataloaders, get_classes_weights
from utils.dataloader import (
    DynamicCollate,
    DynamicAudioTextCollate,
    DynamicAudioTextSpeakerEmbCollate,
    XEUSNestCollate,
    EmbeddingCollate,
    LastLayerEmbeddingCollate
)


class PLWrapper(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.config = config

        self.model = create_ser_model(
            **config.model
        )

        n_classes = config.data.num_classes
        base_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=n_classes),
            "precision": Precision(task="multiclass", average="macro", num_classes=n_classes),
            "recall": Recall(task="multiclass", average="macro", num_classes=n_classes),
            "f1-score": F1Score(task="multiclass", average="macro", num_classes=n_classes),
        })

        self.train_metrics = base_metrics.clone(prefix='train/')
        self.val_metrics = base_metrics.clone(prefix='val/')

        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=n_classes)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = build_dataloaders(self.config)

            # Backward compatibility with previous config versions
            if "loss" not in self.config:
                self.config.loss = {}

            # When using mixup, we use BCEWithLogitsLoss (because we are working with hot-one-encoded targets)
            if self.config.data.get("mixup_alpha", 0.0) > 0.0:
                self.criterion = torch.nn.BCEWithLogitsLoss()
            else:
                if self.config.loss.get("use_weighted_loss", False):
                    print("Using weighted loss")
                    weights = get_classes_weights(self.config).to(self.device)
                else:
                    weights = None
                print(f"Weights: {weights}", self.device)
                if self.config.loss.get("name", "ce").lower() == "ce":
                    print("Using CrossEntropyLoss")
                    self.criterion = torch.nn.CrossEntropyLoss(
                        weight=weights
                    )
                elif self.config.loss.get("name", "ce").lower() == "focal":
                    print("Using Focal Loss")
                    self.criterion = FocalLoss(
                        alpha=self.config.loss.get("alpha", 0.5),
                        gamma=self.config.loss.get("gamma", 2.0),
                        reduction="mean",
                        weight=weights
                    )
                else:
                    raise ValueError(f"Invalid loss: {self.config.loss.name}")

    def compute_class_weights_for_epoch(self, epoch):
        """
        Compute the class weights for the current epoch.
        This function linearly interpolates between the original weights and a uniform distribution.
        """
        original_weights = get_classes_weights(self.config)

        r = float(epoch) / float(self.trainer.max_epochs)
        new_weights = []
        for w in original_weights:
            new_w = (1.0 - r) * w + r * 1.0
            new_weights.append(new_w)
        return torch.tensor(new_weights, dtype=torch.float)

    def train_dataloader(self):
        """Return the training dataloader."""
        print("========= ENTERING TRAIN DATALOADER =========")
        if self.config.model.model_type.lower() == "xeus" or self.config.model.model_type.lower() == "nest":
            collate_fn = XEUSNestCollate()
        elif self.config.model.model_type.lower() == "dynamic":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.model_name)
            collate_fn = DynamicCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
            )
        elif self.config.model.model_type.lower() == "dynamic_audio_text":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.audio_model_name)
            text_tokenizer = AutoTokenizer.from_pretrained(self.config.model.text_model_name)
            collate_fn = DynamicAudioTextCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
                text_tokenizer=text_tokenizer,
            )
        elif self.config.model.model_type.lower() == "dynamic_audio_text_speakeremb":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.audio_model_name)
            text_tokenizer = AutoTokenizer.from_pretrained(self.config.model.text_model_name)
            collate_fn = DynamicAudioTextSpeakerEmbCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
                text_tokenizer=text_tokenizer,
            )
        elif self.config.model.model_type.lower() == "embedding":
            collate_fn = EmbeddingCollate()
        elif self.config.model.model_type.lower() == "last_layer_embedding":
            collate_fn = LastLayerEmbeddingCollate()
        else:
            raise ValueError(f"Invalid model type: {self.config.model.model_type}")


        # Usage of balanced sampling
        if self.config.data.get("use_balanced_sampling", False):
            print("Using Balanced Sampling!")
            # Compute sample weights
            if self.config.data.get("use_balanced_sampling_scheduler", False):
                # Compute the class weights for the current epoch
                class_weights = self.compute_class_weights_for_epoch(self.current_epoch)
                print(f"Class weights: {class_weights}")
            else:
                class_weights = get_classes_weights(self.config)
            # Apply weights to each sample
            sample_weights = [class_weights[t] for t in self.train_dataset.targets]
            sample_weights = torch.tensor(sample_weights, dtype=torch.float)
            # Create the sampler
            weighted_sampler = WeightedRandomSampler(
                weights=sample_weights, # weights here don't need to be normalized to sum to 1 (https://pytorch.org/docs/stable/data.html)
                num_samples=len(sample_weights),
                replacement=True # Sample with replacement (when a sample index is drawn for a row, it is put back in the pool)
            )

            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.train.batch_size,
                sampler=weighted_sampler, # sampler is used instead of shuffle
                num_workers=self.config.train.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=self.config.train.shuffle,
                num_workers=self.config.train.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

    def val_dataloader(self):
        """Return the validation dataloader."""
        if self.config.model.model_type.lower() == "xeus" or self.config.model.model_type.lower() == "nest":
            collate_fn = XEUSNestCollate()
        elif self.config.model.model_type.lower() == "dynamic":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.model_name)
            collate_fn = DynamicCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
            )
        elif self.config.model.model_type.lower() == "dynamic_audio_text":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.audio_model_name)
            text_tokenizer = AutoTokenizer.from_pretrained(self.config.model.text_model_name)
            collate_fn = DynamicAudioTextCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
                text_tokenizer=text_tokenizer,
            )
        elif self.config.model.model_type.lower() == "dynamic_audio_text_speakeremb":
            processor = AutoFeatureExtractor.from_pretrained(self.config.model.audio_model_name)
            text_tokenizer = AutoTokenizer.from_pretrained(self.config.model.text_model_name)
            collate_fn = DynamicAudioTextSpeakerEmbCollate(
                target_sr=self.config.data.target_sr,
                processor=processor,
                text_tokenizer=text_tokenizer,
            )
        elif self.config.model.model_type.lower() == "embedding":
            collate_fn = EmbeddingCollate()
        elif self.config.model.model_type.lower() == "last_layer_embedding":
            collate_fn = LastLayerEmbeddingCollate()
        else:
            raise ValueError(f"Invalid model type: {self.config.model.model_type}")
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        dataset_size = len(dataset)

        # In PyTorch Lightning >= 2.0, trainer.num_devices returns how many devices
        # (GPUs/TPUs/CPUs) the trainer is using.
        gpu_count = self.trainer.num_devices if self.trainer.num_devices else 1

        print("="*100)
        print(f"Dataset size: {dataset_size} | GPU count: {gpu_count}")

        return (dataset_size // gpu_count) * self.trainer.max_epochs

    def configure_optimizers(self):
        """Configures the optimizer and the learning rate scheduler."""
        # Start dataloaders to be able to get the number of steps per epoch
        self.trainer.fit_loop.setup_data()

        max_num_steps = self.num_training_steps()

        print(f"Max number of steps: {max_num_steps}")

        opt_params = self.config.optimizer["params"]
        scheduler_params = self.config.scheduler["params"]

        if self.config.optimizer.name.lower() == "adam":
            optimizer = Adam(
                self.parameters(),
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )
        elif self.config.optimizer.name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer.name}")

        if not self.config["scheduler"]:
            return optimizer

        scheduler = None
        if self.config.scheduler.name.lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=scheduler_params.get("patience", max_num_steps*0.25),
                factor=0.9,
                min_lr=opt_params.get("min_learning_rate", 1.0e-6)
            )

        elif self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1.0e-6),
                lr_max=opt_params["learning_rate"],
                warmup=scheduler_params.get("warmup_lr", max_num_steps*0.05),
                T_max=max_num_steps,
            )

        elif self.config.scheduler.name.lower() == "linearlr":
            scheduler = LinearLR(
                optimizer,
                start_factor=scheduler_params.get("start_factor", 1.0 / 3.0),
                end_factor=scheduler_params.get("end_factor", 1.0),
                total_iters=scheduler_params.get("total_iters", 5),
                last_epoch=scheduler_params.get("last_epoch", -1),
                verbose=scheduler_params.get("verbose", False)
            )
        else:
            raise ValueError(f"Invalid scheduler: {self.config.scheduler.name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def forward(self, x, **kwargs):
        """Forward pass for the model."""
        return self.model(x, **kwargs)

    def on_before_optimizer_step(self, optimizer):
        """
        Used to log the gradients' norm during training.
        Useful for detecting gradient explosion/vanishing.
        """
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def training_step(self, train_batch, batch_idx):
        """Training step."""
        input_features, targets = train_batch

        logits = self.forward(input_features)
        loss = self.criterion(logits, targets)

        if self.config.data.get("mixup_alpha", 0.0) > 0.0:
            targets = torch.argmax(targets, dim=1)

        # Update and log training metrics
        metrics = self.train_metrics(logits, targets)
        # Log metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log training loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        """Validation step."""
        input_features, targets = val_batch

        logits = self.forward(input_features)
        loss = self.criterion(logits, targets)

        # Update and log validation metrics
        metrics = self.val_metrics(logits, targets)

        if self.config.data.get("mixup_alpha", 0.0) > 0.0:
            targets = torch.argmax(targets, dim=1)

        # Log metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log validation loss
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log confusion matrix
        preds = torch.argmax(logits, dim=1)
        self.confusion_matrix(preds, targets)

        return loss

    def on_validation_epoch_end(self):
        # Compute the confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        # Reset confusion matrix for the next epoch
        self.confusion_matrix.reset()
        # Log confusion matrix as a figure
        fig = self._plot_confusion_matrix(cm)
        self.logger.log_image(
            "val/confusion_matrix",
            [wandb.Image(fig, caption="Confusion Matrix")],
            self.current_epoch
        )
        plt.close(fig)

        if self.config.model.get("layer_weight_strategy", "per_layer") == "weighted_sum" or \
            self.config.model.get("layer_weight_strategy", "per_layer") == "weighted_sum_2":
            layer_weights = self.model.get_layer_weights()
            layer_weights = torch.tensor(layer_weights)
            # apply softmax to the weights
            layer_weights = F.softmax(layer_weights, dim=0)
            fig = self._plot_layer_weights_bar_chart(layer_weights)
            self.logger.log_image(
                "val/layer_weights",
                [wandb.Image(fig, caption="Layer Weights")],
                self.current_epoch
            )
            plt.close(fig)

        if self.config.data.get("use_balanced_sampling_scheduler", False):
            class_weights = self.compute_class_weights_for_epoch(self.current_epoch)
            fig = self._plot_class_weights(class_weights)
            self.logger.log_image(
                "val/class_weights",
                [wandb.Image(fig, caption="Class Weights")],
                self.current_epoch
            )
            plt.close(fig)

    def _plot_layer_weights_bar_chart(self, layer_weights):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(layer_weights)), layer_weights)
        ax.set_xticks(range(len(layer_weights)))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Weight")
        ax.set_title("Layer Weights")
        return fig

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")
        return fig

    def _plot_class_weights(self, class_weights):
        original_y_max = get_classes_weights(self.config).max().item()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(class_weights)), class_weights)
        ax.set_xticks(range(len(class_weights)))
        plt.ylim(0, original_y_max)
        ax.set_xlabel("Class")
        ax.set_ylabel("Weight")
        ax.set_title("Class Weights")
        plt.title(f"Epoch {self.current_epoch}")
        return fig