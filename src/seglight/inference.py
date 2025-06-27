import numpy as np
import torch

from seglight.domain import Image


def infer(model,img:Image):
    """
    Runs inference on a single numpy image using the given pytorch model.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model already placed on a device (e.g. `model.to(device)` was
        invoked)
    img : np.ndarray
        Input image as a NumPy array. Expected shape is (H, W) or (H, W, C).

    Returns
    -------
    np.ndarray
        Model prediction as a NumPy array with batch and channel dimensions removed.
        Shape depends on the model output, typically (H, W) or (H, W, C) depending on
        model classes.
    """
    device = model.device
    img = img[None] if len(img.shape) == 2 else np.rollaxis(img, -1)

    img_t = torch.Tensor(img[None]).to(device)
    pred = model(img_t)
    pred = np.squeeze(pred.detach().cpu().numpy())

    if len(pred.shape) == 2:
        return pred
    # channel last
    return np.dstack(pred)
