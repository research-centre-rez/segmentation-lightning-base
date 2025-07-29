---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path

data_path = Path("data")
model_dir = Path("output")
```

# Installing project specific dependencies

These dependencies are not part of the library since it is used by derived applications only. User needs to install then separately.

```python
%pip install huggingface_hub segmentation_models_pytorch matplotlib
```

```python
import zipfile

from huggingface_hub import hf_hub_download

d25_dirpath = hf_hub_download(
    repo_id="research-centre-rez/segmentation_delisa",
    filename="d25.zip",
    repo_type="dataset",
)

with zipfile.ZipFile(d25_dirpath, "r") as zip_ref:
    zip_ref.extractall(data_path)

dataset_path = data_path / "d25"
```

# Dataset formats / Dataset Pairs Loader


```python
import lightning as L
import numpy as np
import torch

import seglight.data as dt


class PrecipitatesSegmentationPairsLoader:
    def __init__(self, cvr_ds: dt.CVRFolderedDSFormat):
        self.ds = cvr_ds

    def _load_dir(self, data_dict):
        imgs = [v["img"] for v in data_dict.values()]
        labels = [np.dstack([v["label"]]) for v in data_dict.values()]
        return imgs, labels

    def load(self, set_type):
        if set_type == "train":
            train_dict = self.ds.load_train()
            return self._load_dir(train_dict)

        if set_type in ("test", "predict"):
            train_dict = self.ds.load_test()
            return self._load_dir(train_dict)

        raise Exception(f"Invalid {set_type=}. Use 'train', 'test' or 'predict'.")


cvr_ds = dt.CVRFolderedDSFormat(dataset_path)
nfa_ogr_data = PrecipitatesSegmentationPairsLoader(cvr_ds)
```

```python
tri, trl = nfa_ogr_data.load("train")
tei, tel = nfa_ogr_data.load("test")
len(tri), len(tei)
```

```python
import albumentations as A


class Augumentations:
    def __init__(
        self,
        patch_size,
        rotate_degrees,
    ):
        self.patch_size = patch_size
        self.rotate_degrees = rotate_degrees

    @property
    def train_augumentation_fn(self):
        # add padding so it can rotate
        square_diagonal_factor = 1.42
        patch_size_padded = int(self.patch_size * square_diagonal_factor)
        transform_list = [
            A.PadIfNeeded(patch_size_padded, patch_size_padded),
            A.CropNonEmptyMaskIfExists(
                height=patch_size_padded,
                width=patch_size_padded,
                ignore_values=None,
                ignore_channels=None,
            ),
            A.Rotate(limit=self.rotate_degrees, interpolation=2),
            A.CenterCrop(self.patch_size, self.patch_size),
        ]

        return A.Compose(transform_list)

    @property
    def val_augumentation_fn(self):
        return A.Compose(
            [
                A.PadIfNeeded(self.patch_size, self.patch_size),
                A.CenterCrop(self.patch_size, self.patch_size),
            ]
        )


aug = Augumentations(128, 15)

```

```python
dataset_train = dt.AugumentedDataset(tri, trl, aug.train_augumentation_fn)
```

```python
dm = dt.TrainTestDataModule(
    nfa_ogr_data,
    augumentations=aug,
    batch_size=32,
    test_batch_size=1,
)

```

# Sanity check

```python
dm.setup("fit")
for bi, bl in dm.val_dl:
    print(bi.shape, bl.shape)
    break

```

# Model 


```python
from segmentation_models_pytorch import Unet

import seglight.training as tr

loss_fn = tr.resolve_loss("dice")


def prepare_model(
    starting_decoder_channel=8, encoder_depth=5, in_channels=1, classes=1
) -> Unet:
    decoder_channels = np.array([starting_decoder_channel] * encoder_depth)
    pows = list(range(encoder_depth))
    decoder_channels = decoder_channels * [2**p for p in pows]
    decoder_channels = decoder_channels.tolist()
    decoder_channels = [int(x) for x in decoder_channels]

    return Unet(
        encoder_name="resnet50",
        in_channels=in_channels,
        classes=classes,
        activation="sigmoid",
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        decoder_use_batchnorm=True,
    )


def model_builder(params, loss_fn) -> tr.SemsegLightningModule:
    """Builds a model based on the provided parameters.
    Args:
        params (dict): A dictionary containing model parameters.
        - decoder_channels (list): A list of decoder channel sizes.
        - encoder_depth (int): The depth of the encoder.
        Returns:
            A Segmentation model.
    """
    decoder_channels = int(params["decoder_channels"])
    encoder_depth = int(params["encoder_depth"])
    m = prepare_model(
        starting_decoder_channel=decoder_channels,
        encoder_depth=encoder_depth,
        in_channels=1,
        classes=1,
    )
    return tr.SemsegLightningModule(m, loss_fn)
```

# Callbacks
- make metrics visible outside
- disable validation progress bar
- save only the best model


```python
# Used to track train/validation metrics
metrics_callback = tr.MetricsCallback()

# disables flickering validation progress bar
no_val_bar_progressbar_cb = tr.NoValBarProgress()

```

<!-- #region -->
# Training with optuna


To monitor training progress in real time, you can launch the Optuna dashboard in your browser using the following command:

```bash
optuna-dashboard sqlite:///optuna_study.db  # Replace with your actual database path if different
```
or install *[Optuna Dashboard](https://marketplace.visualstudio.com/items?itemName=Optuna.optuna-dashboard)* extension for VS code and right click on *.db file and select "Open with Optuna Dashboard".

Important parameters to change
- `n_trials`: Number of trials to run. Each trial is a single training run with different hyperparameters.
- `sampler`: The sampling algorithm used to select hyperparameters. [Here](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) are descriptions of some common samplers.
- `direction`: The optimization direction, either "minimize" or "maximize". This determines whether Optuna is looking for the lowest or highest value of the objective function.
- `metrics_callback`: checkpoint_callback and pruning_callback are by default, optuna needs these to work properly. They do not need to be defined in notebook again.
- `search_space`: The hyperparameter search space. This is a dictionary where keys are hyperparameter names and values are their respective distributions.

<!-- #endregion -->

```python
import torch
from torchmetrics import JaccardIndex

from seglight.hyperparam import (
    OptunaLightningTuner,
    TunerConfig,
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the search space for hyperparameters
search_space = {"decoder_channels": [8, 16], "encoder_depth": [3]}

config = TunerConfig(
    direction="minimize",
    max_epochs=3,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    eval_metrics=JaccardIndex(task="binary"),
    callbacks=[metrics_callback],
    check_val_every_n_epoch=1,
    log_every_n_steps=2,
    model_dir=model_dir,
    study_name="seglight",
)

# Initialize the Optuna Lightning Tuner with config
tuner = OptunaLightningTuner(
    model_builder=model_builder,
    model_class=tr.SemsegLightningModule,
    loss_fn=loss_fn,
    datamodule=dm,
    param_search_space=search_space,
    config=config,
)

study = tuner.run_study(n_trials=2, sampler="grid")

```

```python
print(f"Best trial number: {study.best_trial.number}")
```

<!-- #region -->
# Visualize training metrics

- for other plots visit https://optuna.readthedocs.io/en/stable/reference/visualization/index.html
- or see dashboard in browser
```bash
<!-- #endregion -->

```python
from plotly.io import show
import optuna.visualization


fig = optuna.visualization.plot_intermediate_values(study)

show(fig)
```

# Evaluate model visually

This is a good place to add custom metrics if needed

```python
predictions = tuner.predict(dm, study)
```

```python editable=true slideshow={"slide_type": ""}
import matplotlib.pyplot as plt

# Display the first 5 predictions along with their corresponding images and labels
for (img, l), p in list(zip(dm.pred_dl, predictions, strict=False))[:5]:
    _, axs = plt.subplots(1, 3)

    titles = ["image", "label", "predction"]
    ims = [img, l, p]
    for ax, im, title in zip(axs, ims, titles, strict=False):
        ax.imshow(np.squeeze(im), cmap="gray")
        ax.set_title(title)
    plt.show()
```

```python
# exit()
```
