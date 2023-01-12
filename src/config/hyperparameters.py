# architecture of UNET
UNET_FEATURES = [64, 128, 256, 512]

# training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 6
LOAD_MODEL = False

# image transformation parameters
# original image size is 1918x1280 (reduced for increase training speed)
IMAGE_HEIGHT = 16  # 160
IMAGE_WIDTH = 24  # 240

# data path
TRAIN_IMAGE_DIR = "data/raw/train_images"
TRAIN_MASK_DIR = "data/raw/train_masks"
VAL_IMAGE_DIR = "data/raw/val_images"
VAL_MASK_DIR = "data/raw/val_masks"

# model path
CHECKPOINT_FOLDER = "models"
CHECKPOINT_PATH = "models/checkpoint.pth.tar"
