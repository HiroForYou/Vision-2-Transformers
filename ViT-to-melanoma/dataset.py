import glob
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


def norm01(x):
    return np.clip(x, 0, 255) / 255


def filter_image(p):
    label_data = np.load(p.replace("image", "label"))
    return np.max(label_data) == 1


class Dataset_isic2017(Dataset):
    def __init__(self, split, aug=False, return_only_cls=False):
        super(Dataset_isic2017, self).__init__()

        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.class_csv_path = ""
        self.return_only_cls = return_only_cls

        root_dir = "./processed/isic2017"
        if split == "train":
            self.image_paths = glob.glob(root_dir + "/Train/Image/*.npy")
            self.label_paths = glob.glob(root_dir + "/Train/Label/*.npy")
            self.point_paths = glob.glob(root_dir + "/Train/Point/*.npy")
            self.class_csv_path = root_dir + "/Train/train_cls.csv"
        elif split == "valid":
            self.image_paths = glob.glob(root_dir + "/Validation/Image/*.npy")
            self.label_paths = glob.glob(root_dir + "/Validation/Label/*.npy")
            self.class_csv_path = root_dir + "/Validation/val_cls.csv"
        elif split == "test":
            self.image_paths = glob.glob(root_dir + "/Test/Image/*.npy")
            self.label_paths = glob.glob(root_dir + "/Test/Label/*.npy")
            self.class_csv_path = root_dir + "/Test/test_cls.csv"
        self.image_paths.sort()
        self.label_paths.sort()

        print("Loaded {} frames".format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug

        self.train_transforms = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(),
            ]
        )

        self.valid_transforms = A.Compose([A.Resize(224, 224)])
        self.df_classes = pd.read_csv(self.class_csv_path)
        self.onehot_encode = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def __getitem__(self, index):
        isic_id = self.image_paths[index][-11:-4]
        isic_id = "ISIC_" + isic_id

        image_data = np.load(self.image_paths[index])
        label_data = np.load(self.label_paths[index]) > 0.5

        mask = label_data[..., np.newaxis].astype("uint8")
        # print(mask.shape)
        tsf = (
            self.train_transforms(image=image_data.astype("uint8"), mask=mask)
            if self.aug
            else self.valid_transforms(image=image_data.astype("uint8"), mask=mask)
        )
        image_data, mask_aug = tsf["image"], tsf["mask"]
        label_data = mask_aug[:, :, 0]

        image_data = norm01(image_data)
        label_data = np.expand_dims(label_data, 0)
        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        image_data = image_data.permute(2, 0, 1)

        cls_target = int(
            self.df_classes[self.df_classes["image_id"] == isic_id].melanoma
        )
        # cls_target = np.array([cls_target])
        # cls_target = np.expand_dims(cls_target, 0)
        # cls_target = torch.from_numpy(cls_target).float()
        # cls_target = np.expand_dims(cls_target, 0)

        if self.return_only_cls:
            return {
                "image_path": self.image_paths[index],
                "image": image_data,
                "cls": self.onehot_encode[cls_target],
            }
        else:
            return {
                "image_path": self.image_paths[index],
                "label_path": self.label_paths[index],
                "image": image_data,
                "label": label_data,
                "cls": self.onehot_encode[cls_target],
            }

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    isic = Dataset_isic2017("train")
    isic.__getitem__(1)
