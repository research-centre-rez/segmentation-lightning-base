---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: semseg
    language: python
    name: semseg
---

```python
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path

data_path = Path('data')
model_dir = Path('model')
```

```python
%pip install huggingface_hub
```

```python

import zipfile

from huggingface_hub import hf_hub_download
d25_dirpath = hf_hub_download(
    repo_id='research-centre-rez/segmentation_delisa', 
    filename = 'd25.zip',
    repo_type='dataset'
)

with zipfile.ZipFile(d25_dirpath, 'r') as zip_ref:
    zip_ref.extractall(data_path)

dataset_path = data_path / 'd25'
```

# Dataset/Dataloaders

```python
import cv2
import lightning as L
import numpy as np
import seglight.data as dt
import torch
from seglight.domain import Image

class PrecipitatesSegmentationPairsLoader:
    def __init__(self,cvr_ds:dt.CVRFolderedDSFormat):
        self.ds = cvr_ds

    def _load_dir(self, paths):
        _dict = self.ds.load_dir_dict(paths)
        imgs = [v['img']for v in _dict.values()]
        labels = [ np.dstack([v['label']])  for v in _dict.values()]
        return imgs,labels

    def load(self, set_type):
        train_paths,test_paths = self.ds.read_train_test_paths()

        if set_type == 'train':
            return self._load_dir(train_paths)

        if set_type == 'test' or set_type == 'predict':
            return self._load_dir(test_paths)


        raise Exception(f"Invalid {set_type=}. Use 'train' or 'test'.")

cvr_ds = dt.CVRFolderedDSFormat(dataset_path)
nfa_ogr_data = PrecipitatesSegmentationPairsLoader(cvr_ds)
```

```python
tri,trl = nfa_ogr_data.load('train')
tei,tel = nfa_ogr_data.load('test')
len(tri),len(tei)
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
        # add padding so it cat rotate
        patch_size_padded = int(self.patch_size * 1.42)
        transform_list = [
            A.PadIfNeeded(patch_size_padded, patch_size_padded),
            A.CropNonEmptyMaskIfExists(
                height=patch_size_padded,
                width=patch_size_padded,
                ignore_values=None,
                ignore_channels=None,
                p=1,
            ),
            A.Rotate(limit=self.rotate_degrees, interpolation = 2),
            A.CenterCrop(self.patch_size, self.patch_size),

  #          A.SquareSymmetry(p = .6),
        ]

        return A.Compose(transform_list)

    @property
    def val_augumentation_fn(self):
        return A.Compose([
            A.PadIfNeeded(self.patch_size, self.patch_size),
            A.CenterCrop(self.patch_size, self.patch_size),
        ])
aug = Augumentations(128 ,15)
```

```python
dataset_train = dt.AugumentedDataset(
    tri,
    trl,
    aug.train_augumentation_fn
)
```

```python
dm = dt.TrainTestDataModule(nfa_ogr_data,augumentations=aug, batch_size=32)
dm.setup(None)
```

# Sanity check

```python
for bi,bl in dm.val_dl:
    print(bi.shape,bl.shape)
    break
```

# Model

```python

import seglight.training as tr
from segmentation_models_pytorch import Unet


def prepare_model(
    starting_decoder_channel = 8,
    encoder_depth = 5,
    in_channels = 1,
    classes = 1
):
    decoder_channels = np.array([starting_decoder_channel]*encoder_depth)
    pows = list(range(encoder_depth))
    decoder_channels = decoder_channels * [2**p for p in pows]

    return Unet(
        encoder_name="resnet50",
        in_channels=in_channels,
        classes=classes,
        activation="sigmoid",
        encoder_depth = encoder_depth,
        decoder_channels = decoder_channels,
        decoder_use_batchnorm=True,
    )

m = prepare_model(8,3,1,classes = 1)
loss_fn = tr.resolve_loss("dice")
model = tr.SemsegLightningModule(m,loss_fn)
```

# Callbacks
- make metrics visible outside
- disable validation progress bar
- save only the best model

```python
from lightning.pytorch.callbacks import ModelCheckpoint

metrics_callback = tr.MetricsCallback()
no_val_bar_progressbar_cb = tr.NoValBarProgress()
checkpoint_callback = ModelCheckpoint(
    dirpath=model_dir, 
    save_top_k=2, 
    monitor="val_loss"
)
```

# Training

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    check_val_every_n_epoch=2,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, metrics_callback,no_val_bar_progressbar_cb]
)
trainer.fit(model,datamodule = dm)
```

```python
import matplotlib.pyplot as plt


def read_loss_val(tensor):
    if tensor is None:
        return np.nan

    return tensor.cpu().numpy()

train_loss,val_loss = np.array([(
        read_loss_val(d.get('train_loss')),read_loss_val(d.get('val_loss'))) 
        for d in metrics_callback.metrics
]).T
plt.plot(train_loss,label = 'train')
plt.plot(val_loss,label = 'test')
plt.legend()
```

```python
model.eval()
with torch.no_grad():
    preds_tensors = trainer.predict(model,datamodule=dm)

preds_batched = [p.cpu().numpy() for p in  preds_tensors]
preds = []
for pb in preds_batched:
    for p in pb:
        preds.append(p)
```

```python
import matplotlib.pyplot as plt

for p in preds:
    img = np.squeeze(p)
    plt.imshow(img)
    plt.show()

```
