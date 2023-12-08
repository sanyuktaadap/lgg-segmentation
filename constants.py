import os

# Setup Constants
PATH_PREFIX = './dataset/mri-images'
SPLIT_PATH_PREFIX = './data-split'
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Training Run
RUN_ID = "2"
TRAIN_IMG_PATH = "./data-split/train"
VAL_IMG_PATH = "./data-split/val"
CKPT_PATH = os.path.join("./checkpoints/", RUN_ID)
LOG_PATH = os.path.join("./logs", RUN_ID)
IMG_SIZE = (256, 256)

# Hyperparams
lr = 1e-4
lmbda = 1e-5
batch_size = 16
n_epochs = 40

# Testing
TEST_IMG_PATH = "./data-split/test"
TEST_CKPT_PATH = "./checkpoints/2/checkpoint39.pt"