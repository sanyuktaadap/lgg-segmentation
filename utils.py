from tqdm import tqdm
import os
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torchvision
# Local imports
import constants as C
from metrics import get_metrics

def get_mask_path(path, ext="tif"):
    """_summary_

    Args:
        path (_type_): _description_
        ext (str, optional): _description_. Defaults to "tif".

    Returns:
        _type_: _description_
    """

    name, _ = path.rsplit(".", 1)
    mask = name + f"_mask.{ext}"

    return mask

def copy_file_to_split(folder, idx, split):
    """_summary_

    Args:
        folder (_type_): _description_
        idx (_type_): _description_
        split (_type_): _description_
    """

    img_path = os.path.join(C.PATH_PREFIX, folder, folder + "_" + str(idx) + ".tif")
    mask_path = get_mask_path(img_path)
    dest_path = os.path.join(C.SPLIT_PATH_PREFIX, split)

    os.makedirs(dest_path, exist_ok=True)

    shutil.copy(img_path, dest_path)
    shutil.copy(mask_path, dest_path)

# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#inspect-the-model-using-tensorboard
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def log_img(logger, x, y, y_hat, step, idx, thresh=0.5):
    # convert values of y_hat. if value >= 0.5, makes it 1. others will be 0
    y_hat = (y_hat >= thresh).float()

    c, h, w = x.shape
    y = y.reshape(1, h, w)
    y_hat = y_hat.reshape(1, h, w)
    y = y.repeat(c, 1, 1)
    y_hat = y_hat.repeat(c, 1, 1)

    imgs = torch.stack([x, y, y_hat], dim=0)

    # create grid of images
    img_grid = torchvision.utils.make_grid(imgs)

    # show images
    matplotlib_imshow(img_grid)

    # write to tensorboard
    logger.add_image(f"batch {idx}", img_grid, global_step=step)

def run_epoch(dataloader,
              model,
              device,
              loss_fn,
              logger=None,
              opt=None,
              step=0,
              log_img_factor=5):
    """_summary_

    Args:
        dataloader : torch dataloader that load batches of data
        model : segmentation model
        device : cpu/cuda
        loss_fn : loss function
        logger : to log the performance of the model
        opt : optimizer for gradient decent. Defaults to None.
        step (int, optional): Defaults to 0.

    Returns:
        list: metrics
    """

    epoch_i = step // len(dataloader)

    loss_list = []
    iou_list = []
    dice_list = []
    rec_list = []

    for i, (x, y) in enumerate(tqdm(dataloader)):

        # Moving input to device
        x = x.to(device)
        y = y.to(device)

        # Running forward propagation
        # x (b, c, h, w) -> (b, h, w)
        y_hat = model(x)

        loss = loss_fn(y_hat, y)

        if opt is not None:
            # Make all gradients zero.
            opt.zero_grad()

            # Run backpropagation
            loss.backward()

            # Clipping gradients to 0.01 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            # Update parameters
            opt.step()

        loss_val = loss.item()
        loss_list.append(loss_val)

        # detach removes y_hat from the original computational graph which might be
        # on gpu.
        y_hat = y_hat.detach().cpu()
        y = y.cpu()

        # Compute metrics
        iou = get_metrics(y_hat, y, metric="iou")
        iou_list.append(iou)

        dice = get_metrics(y_hat, y, metric="dice")
        dice_list.append(dice)

        rec = get_metrics(y_hat, y, metric="recall")
        rec_list.append(rec)

        if logger is not None:
            logger.add_scalar(f"loss", loss_val, step)
            logger.add_scalar(f"iou", iou, step)
            logger.add_scalar(f"dice", dice, step)
            logger.add_scalar(f"recall", rec, step)

            if i % log_img_factor == 0:
                j = x.shape[0] // 2
                x = x.cpu()

                log_img(logger, x[j], y[j], y_hat[j], epoch_i, i)

        step += 1

    avg_loss = torch.Tensor(loss_list).mean()
    avg_iou = torch.Tensor(iou_list).mean()
    avg_dice = torch.Tensor(dice_list).mean()
    avg_r = torch.Tensor(rec_list).mean()

    return avg_loss, avg_iou, avg_dice, avg_r
