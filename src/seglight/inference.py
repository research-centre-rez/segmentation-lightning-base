import numpy as np
import torch

import seglight.image_utils as iu
from seglight.domain import Image


def infer(model, img: Image, device="cuda"):
    """
    Runs inference on a single numpy image using the given pytorch model.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model.
    img : np.ndarray
        Input image as a NumPy array. Expected shape is (H, W) or (H, W, C).
    devide : str
        Device str used to move model and image tensor to.
    Returns
    -------
    np.ndarray
        Model prediction as a NumPy array with batch and channel dimensions removed.
        Shape depends on the model output, typically (H, W) or (H, W, C) depending on
        model classes.
    """
    img = img[None] if len(img.shape) == 2 else np.rollaxis(img, -1)
    
    d = next(model.parameters()).device
    if device_name_to_param('cuda') !=  (d.type,str(d.index)):
        model = model.to(device)
        
    img_t = torch.Tensor(img[None]).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(img_t)
    pred = np.squeeze(pred.detach().cpu().numpy())

    if len(pred.shape) == 2:
        return pred
    # channel last
    return np.dstack(pred)


def infer_oversized(
    model,
    img,
    tile_size=2048,
    overlap=256,
    device="cuda",
):
    """
    Runs inference on a single numpy image that using the given pytorch model.
    The image is split into smaller tiles to save memory.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model already placed on a device (e.g. `model.to(device)` was
        invoked)
    img : np.ndarray
        Input image as a NumPy array. Expected shape is (H, W) or (H, W, C).
    tile_size : int
        Size of a tiles the image is split into
    overlap : int
        Overlap of used to blend neighboring tiles.
    devide : str
        Device str used to move model and image tensor to.

    Returns
    -------
    np.ndarray
        Model prediction as a NumPy array with batch and channel dimensions removed.
        Shape depends on the model output, typically (H, W) or (H, W, C) depending on
        model classes.
    """
    tiles, xy = iu.tile_image_with_overlap(img, tile_size, overlap)

    tiles_pred = []
    for tile_img in tiles:
        tile_pred = infer(model, tile_img, device=device)
        tiles_pred.append(tile_pred)

    return iu.blend_tiles(tiles_pred, xy, img.shape)


def device_name_to_param(dev):
    parts = dev.split(':')
    if len(parts) == 1:
        return parts[0],"0"
    elif len(parts) == 2:
        return parts
    else:
        return parts[:2]