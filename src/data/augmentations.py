import albumentations as A
from albumentations.pytorch import ToTensorV2

import src.config.hyperparameters as hp

train_transform = A.Compose(
    [
        A.Resize(height=hp.IMAGE_HEIGHT, width=hp.IMAGE_WIDTH),
        A.Rotate(limit=15, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=hp.IMAGE_HEIGHT, width=hp.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        ),
        ToTensorV2(),
    ],
)
