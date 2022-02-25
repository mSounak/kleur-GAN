import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARENT_DIR = 'data/images/'
MODEL_PATH = 'models/'
LEARNING_RATE = 2e-4
BATCH_SIZE = 12
NUM_WORKERS = 2
IMG_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100.
NUM_EPOCHS = 20
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "models/checkpoints/disc.pth.tar"
CHECKPOINT_GEN = "models/checkpoints/gen.pth.tar"