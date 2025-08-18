import logging
import pathlib

import numpy as np
import torch

import seglight.image_utils as iu
from seglight.domain import Image

logger = logging.getLogger(__name__)


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


def predict_best_trial(datamodule, study, model_dir, threshold=0.5) -> list[np.ndarray]:
    """
    Predicts binary masks for all samples in a given datamodule
    using the best trial model.

    Parameters
    ----------
    datamodule : LightningDataModule
        PyTorch Lightning datamodule containing a `predict_dataloader` method.
    study : optuna.Study
        Optuna study object from which the best trial number is retrieved.
    model_dir : str or pathlib.Path
        Directory where the saved model checkpoints are stored.
    threshold : float, default=0.5
        Threshold applied to model outputs to produce binary masks.

    Returns
    -------
    List[np.ndarray]
        List of binary masks for each sample in the datamodule. Each mask has shape
        (H, W) or (1, H, W) depending on model classes.
    """

    trial_num = study.best_trial.number
    path_of_model = pathlib.Path(model_dir) / f"model_trial_{trial_num}.pt"
    model = torch.jit.load(path_of_model)
    model.eval()
    logger.info(f"Loaded model from {path_of_model} for prediction")

    datamodule.setup("predict")
    loader = datamodule.predict_dataloader()

    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, tuple | list) else batch
            out = model(x)
            binary_mask = (out >= threshold).float()
            preds.append(binary_mask.cpu().numpy())
    logger.info("Prediction completed")
    return [p for b in preds for p in b]
