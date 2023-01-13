import torch
import torch.nn as nn
from loguru import logger
from model import UNET
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config.hyperparameters as hp
from src.data.augmentations import train_transform, val_transform
from src.utils.utils import check_accuracy_dice_score, get_loaders, save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: "torch.nn.modules",
) -> None:
    """Does training for a single epoch with displaying progress bar.

    Parameters
    ----------
    loader : DataLoader
        loader with images and masks
    model : nn.Module
        deep learning model that is trained on images/masks dataset
    optimizer : torch.optim.Optimizer
        optimizer that will update learnable parameters
    loss_function : torch.nn.modules
        loss_function that measures how close model's predictions to true labels
    """
    loop = tqdm(loader, ascii=True)
    for data, targets in loop:
        # preparing batch
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        predictions = model(data)
        loss = loss_function(predictions, targets)

        # backpropagation pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # printing loss value in the tqdm loop
        loop.set_postfix(loss=loss.item())


def main() -> None:
    """Trains model for specified number of epochs."""
    model = UNET(features=hp.UNET_FEATURES, in_channels=3, out_channels=1).to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        hp.TRAIN_IMAGE_DIR,
        hp.TRAIN_MASK_DIR,
        hp.VAL_IMAGE_DIR,
        hp.VAL_MASK_DIR,
        hp.BATCH_SIZE,
        train_transform,
        val_transform,
        hp.NUM_WORKERS,
    )

    best_accuracy = float("-inf")
    for idx in range(hp.NUM_EPOCHS):
        logger.debug("Starting epoch {}", idx)
        training(train_loader, model, optimizer, loss_function)

        # checking accuracy
        accuracy, dice_score = check_accuracy_dice_score(val_loader, model, device=DEVICE)
        logger.debug("Acc: {:.2f}", accuracy * 100)
        logger.debug("Dice score: {:.2f}", dice_score * 100)
        if accuracy > best_accuracy:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state=checkpoint, filename=hp.CHECKPOINT_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    main()
