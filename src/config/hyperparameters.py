# architecture of UNET
UNET_FEATURES = [64, 128, 256, 512]
UNET_RESNET_FEATURES = [(64, 3), (128, 3), (256, 3), (512, 4)] # (num_channels, num_repeats)


# training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

# image transofmation parameters
# original image size is 1918x1280 (reduced for increase training speed)
IMAGE_HEIGHT = 16 # 160
IMAGE_WIDTH = 24 # 240

# data path
TRAIN_IMAGE_DIR = "data/raw/train_images"
TRAIN_MASK_DIR = "data/raw/train_masks"
VAL_IMAGE_DIR = "data/raw/val_images"
VAL_MASK_DIR = "data/raw/val_masks"

# model path
CHECKPOINT_FOLDER = "models"
CHECKPOINT_PATH = "models/checkpoint.pth.tar"
