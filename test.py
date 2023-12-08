import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# Local Imports
from constants import *
from dataset import LGGDataset
from utils import run_epoch

# Load Test data
test_transforms_and_targets = [
    (
        v2.Compose([
            v2.ToImage(),
            v2.Resize(size=IMG_SIZE),
            v2.ToDtype(torch.float, scale=True)
        ]),
        "both"
    ),
    # (v2.ToDtype(torch.float, scale=True), "image"),
    # (v2.ToDtype(torch.uint8, scale=True), "mask")
]

test_dataset = LGGDataset(
    data_path=TEST_IMG_PATH,
    transforms_and_targets=test_transforms_and_targets
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch',
    'unet',
    pretrained=False
)

loss_fn = nn.BCELoss()
test_logger = SummaryWriter(
    log_dir=os.path.join(LOG_PATH, "test")
)

model = model.to(device)

# Testing
model.eval()

test_metrics = run_epoch(
    test_dataloader,
    model,
    device,
    loss_fn,
    test_logger,
    log_img_factor=2
)

print(
    f"Test: Loss - {test_metrics[0]}, " +
    f"IOU - {test_metrics[1]}, " +
    f"Dice - {test_metrics[2]}, " +
    f"Recall - {test_metrics[3]}, "
)