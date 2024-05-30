import argparse
from random import random
from re import A
import comet_ml

from lightning import FastSimLightning
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer

import json
import glob
import random
import torch
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint

import os


parser = argparse.ArgumentParser(description="Train the FastSim model.")
parser.add_argument(
    "-c", "--config", required=True, type=str, help="Path to config file."
)
parser.add_argument(
    "--test_run",
    action="store_true",
    default=False,
    help="No logging, checkpointing, test on a few jets.",
)
parser.add_argument(
    "--gpu", default="", type=str, help="Comma separated list of GPUs to use."
)

if __name__ == "__main__":
    args = parser.parse_args()
    config_path = args.config

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(config_path, "r") as fp:
        config = json.load(fp)

    net = FastSimLightning(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        save_top_k=3,
        save_last=True,
        filename="epoch_{epoch}-val_loss_{val/total_loss:.4f}",
        auto_insert_metric_name=False,
    )

    if args.test_run:
        trainer = Trainer(
            max_epochs=config["num_epochs"],
            accelerator="gpu" if args.gpus != "" else None,
            devices=1 if args.gpus != "" else None,
            callbacks=[checkpoint_callback],
            fast_dev_run=True,
        )
    else:
        comet_logger = CometLogger(
            api_key=os.environ["COMET_API_KEY"],
            save_dir="logs",
            project_name=os.environ["COMET_PROJECT_NAME"],
            workspace=os.environ["COMET_WORKSPACE"],
            experiment_name=config["name"],
        )

        net.set_comet_exp(comet_logger.experiment)
        comet_logger.experiment.log_asset(config_path, file_name="config")

        all_files = glob.glob("./*.py") + glob.glob("models/*.py")
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath)

        trainer = Trainer(
            max_epochs=config["num_epochs"],
            accelerator="gpu" if args.gpus != "" else None,
            devices=1 if args.gpus != "" else None,
            logger=comet_logger,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(net, ckpt_path=config["resume_from_checkpoint"])
