import os

import albumentations as A
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.config.hyperparameters import CHECKPOINT_FOLDER
from src.data.dataset import SegmentationDataset


def save_checkpoint(state: dict, filename: str) -> None:
    """Save current state of the model with parameters.

    Parameters
    ----------
    state : dict
        parameters of the model represented as a dict
    filename : str
        name of the file in which model will be saved
    """
    logger.debug("Saving checkpoint")
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(checkpoint: dict, model: torch.nn.Module) -> None:
    """Load saved state of the model with parameters.

    Parameters
    ----------
    checkpoint : dict
        dictionary with value for each model's parameter
    model : torch.nn.Module
        model in which saved parameter values will be loaded
    """
    logger.debug("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir: str,
    train_mask_dir: str,
    val_dir: str,
    val_mask_dir: str,
    batch_size: int,
    train_transform: A.Compose,
    val_transform: A.Compose,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Return train and validation data loaders.

    Parameters
    ----------
    train_dir : str
        path to train images
    train_mask_dir : str
        path to train masks
    val_dir : str
        path to validation images
    val_mask_dir : str
        path to validation masks
    batch_size : int
        the size of the batch
    train_transform : A.Compose
        set of albumentations transformations that will be applied
        to images and masks from training set
    val_transform : A.Compose
        set of albumentations transformations that will be applied
        to images and masks from validation set
    num_workers : int
        number of parallel workers

    Returns
    -------
    tuple[DataLoader, DataLoader]
        train and validation data loaders
    """
    train_ds = SegmentationDataset(
        images_dir=train_dir,
        masks_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_ds = SegmentationDataset(
        images_dir=val_dir,
        masks_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy_dice_score(loader: DataLoader, model: torch.nn.Module, device: str = "cuda") -> tuple[float, float]:
    """Returns accuracy and dice score.

    Parameters
    ----------
    loader : DataLoader
        loader with images and masks
    model : torch.nn.Module
        U-Net model
    device : str, optional
        on which device model and data a stored, by default "cuda"

    Returns
    -------
    tuple[float, float]
        accuracy and dice scores
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    accuracy = num_correct / num_pixels
    dice_score /= len(loader)

    model.train()

    return accuracy, dice_score
