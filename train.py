import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

# Local Imports
from constants import *
from dataset import LGGDataset
from utils import run_epoch

# Load Train data
train_transforms_and_targets = [
    (
        v2.Compose([
            v2.ToImage(),
            v2.Resize(size=IMG_SIZE),
            v2.RandomRotation(degrees=20),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToDtype(torch.float, scale=True)
        ]),
        "both"
    ),
    # (v2.ToDtype(torch.float, scale=True), "image"),
    # (v2.ToDtype(torch.uint8, scale=True), "mask")
]

train_dataset = LGGDataset(
    data_path=TRAIN_IMG_PATH,
    transforms_and_targets=train_transforms_and_targets
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

# Load Val data
val_transforms_and_targets = [
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

val_dataset = LGGDataset(
    data_path=VAL_IMG_PATH,
    transforms_and_targets=val_transforms_and_targets
)

val_dataloader = DataLoader(
    val_dataset,
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
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda) # weight decay is equivalent of L2 regularization in Adam
train_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "train"))
val_logger = SummaryWriter(log_dir=os.path.join(LOG_PATH, "val"))

model = model.to(device)

global_step = 0

os.makedirs(CKPT_PATH, exist_ok=True)

# Logging model graph
x, _ = next(iter(train_dataloader))
train_logger.add_graph(model, x.to(device))

# Training
for i in range(n_epochs):
    model.train()
    # run one epoch of training
    train_metrics = run_epoch(
        train_dataloader,
        model,
        device,
        loss_fn,
        train_logger,
        opt=opt,
        step=i*len(train_dataloader),
        log_img_factor=15
    )

    model.eval()
    # run one epoch of validation
    val_metrics = run_epoch(
        val_dataloader,
        model,
        device,
        loss_fn,
        val_logger,
        step=i*len(val_dataloader),
        log_img_factor=2
    )

    print(f"Epoch {i}:")
    print(
        f"Train: Loss - {train_metrics[0]}, " +
        f"IOU - {train_metrics[1]}, " +
        f"Dice - {train_metrics[2]}, " +
        f"Recall - {train_metrics[3]}, "
    )

    print(
        f"Val: Loss - {val_metrics[0]}, " +
        f"IOU - {val_metrics[1]}, " +
        f"Dice - {val_metrics[2]}, " +
        f"Recall - {val_metrics[3]}, "
    )

    # Logging per epoch so that traning and val can be compared properly
    # Because of equal no. of data points on the graph
    train_logger.add_scalar(f"epoch/loss", train_metrics[0], i)
    train_logger.add_scalar(f"epoch/iou", train_metrics[1], i)
    train_logger.add_scalar(f"epoch/dice", train_metrics[2], i)
    train_logger.add_scalar(f"epoch/recall", train_metrics[3], i)

    val_logger.add_scalar(f"epoch/loss", val_metrics[0], i)
    val_logger.add_scalar(f"epoch/iou", val_metrics[1], i)
    val_logger.add_scalar(f"epoch/dice", val_metrics[2], i)
    val_logger.add_scalar(f"epoch/recall", val_metrics[3], i)

    torch.save(model.state_dict(), os.path.join(CKPT_PATH, f"checkpoint{i}.pt"))