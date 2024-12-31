import os
import json
import argparse
from pprint import pprint

import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.pl_wrapper import PLWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
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
    # Only use this argument if you want to use multiple GPUs with DDP
    parser.add_argument(
        "-gpus",
        "--gpus",
        required=False,
        type=json.loads,
        default=None
    )
    parser.add_argument(
        "-ck",
        "--checkpoint-dir",
        required=False,
        type=str,
        default="../checkpoints/ser-2025",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    exp_title = config.title

    tags = ["InterSpeech-SER-2025"]
    tags += config.tags  # add tags defined for experiments

    wandb.init(
        project="InterSpeech-SER-2025",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True),
    )
    logger = WandbLogger(
        project="InterSpeech-SER-2025",
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True),
    )

    config["model_checkpoint"].pop("dirpath")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]

    model = PLWrapper(config)

    pprint(model)

    if args.gpus is not None:
        print("Using multiple GPUs!")
        print(type(args.gpus), args.gpus)
        trainer = pl.Trainer(
            **config["trainer"],
            logger=logger,
            callbacks=callbacks,
            devices=args.gpus,
            strategy="ddp",
            default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
        )
    else:
        trainer = pl.Trainer(
            **config["trainer"],
            logger=logger,
            callbacks=callbacks,
            devices=[args.gpu],
            default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
        )

    trainer.fit(model)


if __name__ == "__main__":
    main()
