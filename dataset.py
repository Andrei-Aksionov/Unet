import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_filenames = os.listdir(self.images_dir)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        # for dataset I used karvana dataset from kaggle competition. In this dataset
        # masks have the same name as images only with _mask postfix and different extension
        mask_path = os.path.join(self.masks_dir, self.image_filenames[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path).convert("L"))
        # as a double check binarizing mask
        mask = np.where(mask < 0.5, 0, 1)

        # for augmentation `albumentations` is used
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
