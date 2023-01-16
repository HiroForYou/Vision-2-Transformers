import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def process_isic(
    dim=(352, 352),
    save_dir="./processed/isic2016",
    image_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_Data",
    mask_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_GroundTruth",
    save_as_npy=True,
):

    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)

    image_path_list = list(filter(lambda x: x[-3:] == "jpg", image_path_list))
    mask_path_list = list(filter(lambda x: x[-3:] == "png", mask_path_list))

    image_path_list.sort()
    mask_path_list.sort()

    print(len(image_path_list), len(mask_path_list))

    # ISBI Dataset
    for image_path, mask_path in tqdm(zip(image_path_list, mask_path_list)):
        if image_path[-3:] == "jpg":
            # print(image_path)
            assert (
                os.path.basename(image_path)[:-4].split("_")[1]
                == os.path.basename(mask_path)[:-4].split("_")[1]
            )
            _id = os.path.basename(image_path)[:-4].split("_")[1]
            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            image_new = np.array(image_new, dtype=np.uint8)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)
            mask_new = cv2.blur(mask_new, (3, 3))
            mask_new = cv2.cvtColor(mask_new, cv2.COLOR_BGR2GRAY)
            # mask_new = np.array(mask_new, dtype=np.uint8)

            save_dir_path_image = save_dir + "/Image"
            os.makedirs(save_dir_path_image, exist_ok=True)

            save_dir_path_mask = save_dir + "/Label"
            os.makedirs(save_dir_path_mask, exist_ok=True)

            # #print(image_new.shape)
            if save_as_npy:
                np.save(os.path.join(save_dir_path_image, _id + ".npy"), image_new)
                np.save(os.path.join(save_dir_path_mask, _id + ".npy"), mask_new)
            else:
                cv2.imwrite(
                    os.path.join(save_dir_path_image, "ISIC_" + _id + ".jpg"), mask_new
                )
                cv2.imwrite(
                    os.path.join(save_dir_path_mask, "ISIC_" + _id + ".jpg"), image_new
                )


def process_ph2():
    PH2_images_path = "/data2/cf_data/skinlesion_segment/PH2_rawdata/PH2_Dataset_images"

    path_list = os.listdir(PH2_images_path)
    path_list.sort()

    for path in path_list:
        image_path = os.path.join(
            PH2_images_path, path, path + "_Dermoscopic_Image", path + ".bmp"
        )
        label_path = os.path.join(
            PH2_images_path, path, path + "_lesion", path + "_lesion.bmp"
        )
        image = plt.imread(image_path)
        label = plt.imread(label_path)
        label = label[:, :, 0]

        dim = (512, 512)
        image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

        image_save_path = os.path.join(
            "/data2/cf_data/skinlesion_segment/PH2_rawdata/PH2/Image", path + ".npy"
        )
        label_save_path = os.path.join(
            "/data2/cf_data/skinlesion_segment/PH2_rawdata/PH2/Label", path + ".npy"
        )

        np.save(image_save_path, image_new)
        np.save(label_save_path, label_new)


if __name__ == "__main__":
    process_isic(
        dim=(512, 512),
        save_dir="./processed/isic2016/Train",
        image_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_Data",
        mask_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_GroundTruth",
    )
    process_isic(
        dim=(512, 512),
        save_dir="./processed/isic2016/Test",
        image_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_Data",
        mask_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_GroundTruth",
    )
    process_isic(
        dim=(512, 512),
        save_dir="./processed/isic2016/Validation",
        image_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_Data",
        mask_dir_path="./dataset/isic2016/ISBI2016_ISIC_Part1_Training_GroundTruth",
    )
