import torch

import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from lightning.pytorch.utilities import grad_norm
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection

from models.factory import create_ser_model
from utils.schedulers import CosineWarmupLR, LinearLR


class PLWrapper(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.config = config

        self.model = create_ser_model(
            **config.model
        )
        # When using mixup, we use BCEWithLogitsLoss (because we are working with hot-one-encoded targets)
        if config.data.mixup_alpha > 0.0:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        n_classes = config.data.num_classes
        base_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=n_classes),
            "precision": Precision(task="multiclass", average="macro", num_classes=n_classes),
            "recall": Recall(task="multiclass", average="macro", num_classes=n_classes),
            "f1-score": F1Score(task="multiclass", average="macro", num_classes=n_classes),
        })

        self.train_metrics = base_metrics.clone(prefix='train/')
        self.val_metrics = base_metrics.clone(prefix='val/')

    def configure_optimizers(self):
        """Configures the optimizer and the learning rate scheduler."""
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
                patience=scheduler_params.get("patience", self.trainer.max_steps*0.25),
                factor=0.9,
                min_lr=opt_params.get("min_learning_rate", 1.0e-6)
            )

        elif self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1.0e-6),
                lr_max=opt_params["learning_rate"],
                warmup=scheduler_params.get("warmup_lr", self.trainer.max_steps*0.05),
                T_max=self.trainer.max_steps
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

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
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

        if self.config.data.mixup_alpha > 0.0:
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

        if self.config.data.mixup_alpha > 0.0:
            targets = torch.argmax(targets, dim=1)

        # Log metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log validation loss
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss