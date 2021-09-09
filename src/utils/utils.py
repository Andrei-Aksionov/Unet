import os

import albumentations as A
import torch
from src.config.hyperparameters import CHECKPOINT_FOLDER
from src.data.dataset import SegmentationDataset
from torch.utils.data import DataLoader


def save_checkpoint(state: dict, filename: str) -> None:
    print("Saving checkpoint")
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(checkpoint: dict, model: torch.nn.Module) -> None:
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform: A.Compose,
    val_transform: A.Compose,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SegmentationDataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SegmentationDataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy_dice_score(loader: DataLoader, model: torch.nn.Module, device: str = "cuda") -> tuple[float, float]:
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
