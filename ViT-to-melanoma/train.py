import argparse

import numpy as np
import torch
from configs import CONFIGS as CONFIGS_ViT_seg
from model import VisionTransformer as ViT_seg
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
    "--n_classes", type=int, default=1, help="output channel of network"
)
parser.add_argument(
    "--patience", type=int, default=15, help="patience to early stoping"
)
parser.add_argument(
    "--max_epochs", type=int, default=30, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=6, help="batch_size per gpu")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument(
    "--base_lr", type=float, default=0.02, help="segmentation network learning rate"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument(
    "--n_skip", type=int, default=0, help="using number of skip-connect, default is num"
)
parser.add_argument(
    "--vit_name", type=str, default="ViT-B_16", help="select one vit model"
)
parser.add_argument(
    "--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16"
)
parser.add_argument(
    "--scheduler", type=str, default="CosineAnnealingLR", help="LR scheluler"
)
parser.add_argument(
    "--train_mode",
    type=str,
    default="seg_cls_init",
    help="Train mode (seg_cls_init, seg_cls_middle, seg_cls_masking, seg)",
)
parser.add_argument("--optimizer", type=str, default="sgd", help="adam, sgd")
args = parser.parse_args()


if __name__ == "__main__":

    seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    args.num_classes = 1
    args.is_pretrain = True
    snapshot_path = dataset_name + "_" + str(args.img_size)
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path += "_" + args.vit_name + "_" + args.train_mode
    snapshot_path = snapshot_path + "_skip" + str(args.n_skip)
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

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.n_classes
    config_vit.base_lr = args.base_lr
    config_vit.n_skip = args.n_skip
    config_vit.skip_channels = None
    config_vit.batch_size = args.batch_size
    config_vit.scheduler = args.scheduler
    config_vit.min_lr = 1e-6
    config_vit.T_max = 800
    config_vit.T_0 = 500
    config_vit.train_mode = args.train_mode
    config_vit.optimizer = args.optimizer

    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    model = ViT_seg(config_vit, img_size=args.img_size)
    model.load_from(weights=np.load(config_vit.pretrained_path))

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_dice_score_avg",
        mode="max",
        dirpath="./weights",
        filename=snapshot_path
        + "-{epoch:02d}-{val_dice_score_avg:02f}-{val_acc_cls:02f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_dice_score_avg",
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
