import os
from PIL import Image
from torch.utils.data import Dataset
from utils import get_mask_path
import numpy as np

class LGGDataset(Dataset):
    """LGGDataset
    Args:
        Dataset : Parent torch dataset class
    """
    def __init__(
        self,
        data_path,
        transforms_and_targets=None,
    ):
        """_summary_

        Args:
            data_path (_type_): _description_
            transforms_and_targets (_type_, optional): _description_. Defaults to None.
        """

        # Path to training data
        self.data_path = data_path
        self.transforms_and_targets = transforms_and_targets
        # List containing paths to all the images
        self.imgs_path = []

        # List of all folders inside data_path
        file_names = os.listdir(self.data_path)

        # Retrieve each image (.tif) in folders
        for name in file_names:
            # only keeping track of image paths
            if "mask" in name:
                continue

            path = os.path.join(self.data_path, name)
            self.imgs_path.append(path)

    def __len__(self):
        """Gets the length of the dataset
        Returns:
            int: total number of data points
        """
        return len(self.imgs_path)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """

        img_path = self.imgs_path[idx]
        mask_path = get_mask_path(img_path)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transforms_and_targets is not None:
          for transforms, target in self.transforms_and_targets:
              if target == "image":
                  img = transforms(img)
              elif target == "mask":
                  mask = transforms(mask)
              elif target == "both":
                  img, mask = transforms(img, mask) ## add to notes

        return img, mask