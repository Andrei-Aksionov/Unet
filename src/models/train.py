import src.config.hyperparameters as hp
import torch
import torch.nn as nn
from albumentations import augmentations
from src.data.augmentations import train_transform, val_transform
from src.utils.utils import (check_accuracy, get_loaders, load_checkpoint,
                             save_checkpoint, save_predictions_as_imgs)
from torch._C import device
from torch.serialization import load
from tqdm import tqdm

from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training(loader, model, optimizer, loss_function, scaler):
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


def main():
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

    # TODO: Tasks pending completion -@andreiaksionov at 9/4/2021, 7:31:51 PM
    # Implement evaluation function

    scaler = torch.cuda.amp.GradScaler()
    for idx in range(hp.NUM_EPOCHS):
        print(f"Starting epoch: {idx}")
        training(train_loader, model, optimizer, loss_function, scaler)

        # saving model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state=checkpoint, filename=hp.CHECKPOINT_PATH)

        # checking accuracy
        check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()
