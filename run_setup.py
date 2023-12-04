# Libraries
import os
import numpy as np
import math
from tqdm import tqdm
# Local imports
import constants as C
from utils import copy_file_to_split

img_folders = os.listdir(C.PATH_PREFIX)

for i, folder in enumerate(tqdm(img_folders)):
    img_paths = os.listdir(os.path.join(C.PATH_PREFIX, folder))
    # files are 1-indexed
    imgs_idx = list(range(1, len(img_paths) // 2 + 1))
    np.random.shuffle(imgs_idx)
    train_len = math.floor(C.TRAIN_SPLIT * len(imgs_idx))
    val_len = math.floor(C.VAL_SPLIT * len(imgs_idx))

    train_idxs, val_idxs, test_idxs = (
        imgs_idx[: train_len],
        imgs_idx[train_len: train_len + val_len],
        imgs_idx[train_len + val_len:]
    )

    for num in train_idxs:
        copy_file_to_split(folder, num, split="train")

    for num in val_idxs:
        copy_file_to_split(folder, num, split="val")

    for num in test_idxs:
        copy_file_to_split(folder, num, split="test")

print("Train Set:", len(os.listdir(os.path.join(C.SPLIT_PATH_PREFIX, "train"))))
print("Val Set:", len(os.listdir(os.path.join(C.SPLIT_PATH_PREFIX, "val"))))
print("Test Set:", len(os.listdir(os.path.join(C.SPLIT_PATH_PREFIX, "test"))))