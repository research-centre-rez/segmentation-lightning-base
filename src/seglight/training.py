import copy

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.functional import F
from torch.nn import BCELoss, Module

import seglight.image_utils as iu


class SemsegLightningModule(L.LightningModule):
    def __init__(self, model, loss_fn, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = learning_rate

    def _step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        return self.loss_fn(outputs, targets)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        loss = self._step(batch)
        self.log("test_loss", loss)

    def predict_step(self, batch):
        imgs, _ = batch
        # fixed as long as images have sensible dimensions
        pad_stride = 32

        padded, pads = iu.pad_to(imgs, pad_stride)
        res_tensor = self.model(padded)
        return iu.unpad(res_tensor, pads)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def resolve_loss(loss_name, **kwargs):
    match loss_name:
        case "bce":
            return BCELoss(**kwargs)
        case "dice":
            return DiceLoss(**kwargs)
        case "fl":
            alpha = kwargs.get("alpha", 0.9)
            gamma = kwargs.get("gamma", 0.8)
            return FocalLoss(alpha=alpha, gamma=gamma)
        case _:
            raise ValueError(f"invalid loss {loss_name=}")


class DiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(Module):
    def __init__(
        self,
        alpha=0.8,
        gamma=2,
        reduction="mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        bce_exp = torch.exp(-bce)
        return self.alpha * (1 - bce_exp) ** self.gamma * bce


# Utils Callbacks used in the example. See notebooks for more detail or
# visit docs here
# https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
class MetricsCallback(L.Callback):
    """
    A PyTorch Lightning callback to store validation metrics after
    each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, _):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)


class NoValBarProgress(TQDMProgressBar):
    """
    A custom progress bar callback that disables the validation progress bar in
    PyTorch Lightning.

    Inherits from TQDMProgressBar and overrides methods related to validation
    progress display.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def on_validation_batch_end(self, *args, **kwargs):
        # Override to do nothing (disables validation batch bar)
        pass
