---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: computer-vision
    language: python
    name: .venv
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

# Installing project specific dependencies

These dependencies are not part of the library since it is used by derived applications only. User needs to install then separately.

```python
%pip install huggingface_hub segmentation_models_pytorch matplotlib
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

<!-- #region -->
# Dataset formats / Dataset Pairs Loader

The input dataset can have any format (CAMVID 1.0, CVAT for IMAGES, Citiescapes). Such data can be loaded and manipulated with if needed. The loading and manipulation steps were split into separate classes to allow for easy combing of data.


## CVR Dataset

The aim of the format is to read any data structure and unify it's in-memory format. In this example, I use the following structure of a single class binary semantic segmentation segmentation dataset:

```
- root
  - test.txt
  - sample_1
    - img.png
    - label.png
  - sample_2
    - img.png
    - label.png
...
  - sample_n
    - img.png
    - label.png
```

Images `img.png` and `label.png` are same size images. The file `test.txt` conains name of the samples which should be used for testing e.g.
```bash
$ cat test.txt
sample_2
sample_3
sample_79
```

The class `CVRFolderedDSFormat` loads this data into a dictionary (see `load_train`/`load_test` methods) in a form
```
{
  "sample_1": {
    "img": IMAGE_ARRAY,
    "label": IMAGE_ARRAY
  }
  ...
}
```
The `IMAGE_ARRAY` is a numpy array of float32 representing the image data. In multiclass segmentations, you can have multiple labels defined e.g:
```
{
  "sample_1": {
    "img": IMAGE_ARRAY,
    "class_1": IMAGE_ARRAY,
    ...
    "class_n": IMAGE_ARRAY,
  }
}
```

The naming could vary as long as the pairs loader understands it. The convention is that the
- image is named `img.png`,
- single class binary image is 'label.png'
- multi class is custom

Should you need data from aforementioned CAMVID or other dataset format, you need to write this part on it's own. See `data.CVRFolderedDSFormat` for more detail.

## Data Loader e.g. Pairs Loader

Pairs Loader is the class responsible for transforming in-memory data into a training pair of image and its label. In this example, this task is trivial and the `PrecipitatesSegmentationPairsLoader` is trivial. However, should you need to combine more classes into a single label, you would need to stack multiple images into one. For example, if this dataset contained one more class `artifacts` for anything that is not supposed to be on the image (scratch, corrosion pit ) you would need to reflect this logic in the `PairsLoader` class with something like this: `labels = [np.dstack([v['label'].v['artifacts']]) for v in data_dict.values()]. 

Additionally, you can calculate your own labels and add them here. For example borders that can be defined as `border = label_dilated - label`, where `label_dilated` is morphologically dilated image. Again, this need to be reflected in the implementation:

```
borders = [ dilate(v['label']) - v['label'] for v in data_dict.values()]
labels = [np.dstack([v['label'].v['artifacts'],b]) for v,b in zip(data_dict.values(),borders)]
```
<!-- #endregion -->

```python
import lightning as L
import numpy as np
import torch

import seglight.data as dt


class PrecipitatesSegmentationPairsLoader:
    def __init__(self,cvr_ds:dt.CVRFolderedDSFormat):
        self.ds = cvr_ds

    def _load_dir(self, data_dict):
        imgs = [v['img']for v in data_dict.values()]
        labels = [ np.dstack([v['label']])  for v in data_dict.values()]
        return imgs,labels

    def load(self, set_type):
        if set_type == 'train':
            train_dict = self.ds.load_train()
            return self._load_dir(train_dict)

        if set_type in ('test', 'predict'):
            train_dict = self.ds.load_test()
            return self._load_dir(train_dict)

        raise Exception(f"Invalid {set_type=}. Use 'train', 'test' or 'predict'.")

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
        # add padding so it can rotate
        square_diagonal_factor = 1.42
        patch_size_padded = int(self.patch_size * square_diagonal_factor)
        transform_list = [
            A.PadIfNeeded(patch_size_padded, patch_size_padded),
            A.CropNonEmptyMaskIfExists(
                height=patch_size_padded,
                width=patch_size_padded,
                ignore_values=None,
                ignore_channels=None
            ),
            A.Rotate(limit=self.rotate_degrees, interpolation = 2),
            A.CenterCrop(self.patch_size, self.patch_size),
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
dm = dt.TrainTestDataModule(
    nfa_ogr_data,
    augumentations=aug,
    batch_size=32,
    test_batch_size=1,
)
```

# Sanity check
This step is here to check that everything is properly setup. If not, the error will be seen here and now and not in the training loop.

```python
dm.setup('fit')
for bi,bl in dm.val_dl:
    print(bi.shape,bl.shape)
    break
```

# Model

```python

from segmentation_models_pytorch import Unet

import seglight.training as tr


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

# Used to track train/validation metrics
metrics_callback = tr.MetricsCallback()
# disables flickering validation progress bar
no_val_bar_progressbar_cb = tr.NoValBarProgress()

# checkpoints the best model based on a monitored `val_loss`
# this is crutial for prediction/test steps of trainer since
# it uses this to select the best model to use for prediction/testing
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

# Visualize training metrics

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

# Evaluate model visually

This is a good place to add custom metrics if needed

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

for (img,l),p in zip(dm.pred_dl,preds, strict=False):
    _,axs = plt.subplots(1,3)

    titles = ['image','label','predction']
    ims = [img,l,p]
    for ax,im,title in zip(axs,ims,titles, strict=False):
        ax.imshow(np.squeeze(im),cmap='gray')
        ax.set_title(title)
    plt.show()

```

```python

```
