import argparse

import ml_collections
import torch
from model import BinaryClassifierLightning
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="isic2017", help="experiment_name")
parser.add_argument(
    "--patience", type=int, default=15, help="patience to early stoping"
)
parser.add_argument(
    "--max_epochs", type=int, default=30, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=32, help="batch_size per gpu")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--base_lr", type=float, default=0.02, help="network learning rate")
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument(
    "--vit_name", type=str, default="deit_tiny", help="select one vit model"
)
parser.add_argument(
    "--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16"
)
parser.add_argument(
    "--scheduler", type=str, default="CosineAnnealingLR", help="LR scheluler"
)
parser.add_argument("--optimizer", type=str, default="sgd", help="adam, sgd")
args = parser.parse_args()


if __name__ == "__main__":

    seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    args.is_pretrain = True
    snapshot_path = dataset_name + "_" + str(args.img_size)
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path = (
        snapshot_path + "_vitpatch" + str(args.vit_patches_size)
        if args.vit_patches_size != 16
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_max_epochs" + str(args.max_epochs)
        if args.max_epochs != 30
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = (
        snapshot_path + "_lr" + str(args.base_lr)
        if args.base_lr != 0.01
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = (
        snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    )

    config_vit = ml_collections.ConfigDict()
    config_vit.base_lr = args.base_lr
    config_vit.scheduler = args.scheduler
    config_vit.batch_size = args.batch_size
    config_vit.min_lr = 1e-6
    config_vit.T_max = 800
    config_vit.T_0 = 500
    config_vit.optimizer = args.optimizer

    model = BinaryClassifierLightning(config=config_vit)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_acc_cls",
        mode="max",
        dirpath="./weights",
        filename=snapshot_path + "-{epoch:02d}-{val_acc_cls:02f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc_cls",
        min_delta=1e-5,
        patience=args.patience,
        verbose=False,
        mode="max",
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./logs/")

    trainer = Trainer(
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.max_epochs,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            lr_monitor_callback,
            TQDMProgressBar(refresh_rate=8),
            ModelSummary(max_depth=1),
        ],
        logger=[tb_logger],
        precision=16,
    )

    # trainer.validate(model)
    trainer.fit(model)

    # trainer.test(ckpt_path=checkpoint_callback.best_model_path)
