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
import seglight.data as dt
import seglight.seg_io as sio
from pathlib import Path
```

```python
dataset_cvr = dt.CVRFolderedDSFormat("data/ogr/")
dataset_destination = Path('/tmp/camvid_test_ii')
```

```python
label_color_mapping = {
    "rods": (255, 0, 247),
    "grids": (0, 255, 47)
}

cvr_dataset = dt.CVRFolderedDSFormat("/home/jry/data/ogr/")

dt.camvid_from_cvr(
    dataset_destination,
    cvr_dataset,
    label_color_mapping,
)
```

# Validation

```python
import datumaro as dm

ds = dm.Dataset.import_from(str(dataset_destination.absolute()))
ds
```
