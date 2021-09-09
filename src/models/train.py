import src.config.hyperparameters as hp
import torch
import torch.nn as nn
from src.data.augmentations import train_transform, val_transform
from src.utils.utils import check_accuracy_dice_score, get_loaders, save_checkpoint
from tqdm import tqdm

from torch.utils.data import DataLoader
from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training(loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_function, scaler) -> None:
    loop = tqdm(loader)

    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_function(predictions, targets)

        # backpropagation pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # printing loss value in the tqdm loop
        loop.set_postfix(loss=loss.item())


def main() -> None:
    model = UNET(features=hp.UNET_FEATURES, in_channels=3, out_channels=1).to(DEVICE)
    # TODO: Tasks pending completion -@andreiaksionov at 9/4/2021, 7:00:56 PM
    # change to cross entropy loss
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
        hp.PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    best_accuracy = float("-inf")
    for idx in range(hp.NUM_EPOCHS):
        print(f"Starting epoch: {idx}")
        training(train_loader, model, optimizer, loss_function, scaler)

        # checking accuracy
        accuracy, dice_score = check_accuracy_dice_score(val_loader, model, device=DEVICE)
        print(f"Acc: {accuracy*100:.2f}")
        print(f"Dice score: {dice_score*100:.2f}")
        if accuracy > best_accuracy:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state=checkpoint, filename=hp.CHECKPOINT_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    main()
