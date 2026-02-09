import imageio
from abc import abstractmethod, ABC
import torch
import zarr
import numpy as np
import numpy.typing as npt
import tifffile as tiff

from dataclasses import dataclass
from functools import lru_cache
import os
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
##from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union, Dict, Any, List
import seglight.seg_io as sio

ColorRGB = tuple[int,int,int]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

import sklearn.model_selection as ms



class BaseDictLoader(Dataset,ABC):

    @abstractmethod
    def split_train_val(self,validation_percentage, rng = None)->tuple[Dataset,Dataset]:
        pass


class CamVidImgDictLoader(BaseDictLoader):
    def __init__(
        self,
        image_paths = Sequence[os.PathLike],
        label_paths = Sequence[os.PathLike],
        label_mapping = dict[str,ColorRGB],
        assume_grayscale = True,
        name_mapping = None
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_mapping = label_mapping
        self.assume_grayscale = assume_grayscale
        self.name_mapping = name_mapping

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        if self.assume_grayscale:
            img = sio.imread_as_float(img_path)
        else:
            img = sio.imread_raw(img_path)
            
        lbl_ch = sio.imread_raw(label_path)

        lbls = _colors_to_masks(lbl_ch, self.label_mapping)

        imgs = {"img": img} | lbls
        return _remap_names(imgs, self.name_mapping)

    def split_train_val(self,validation_percentage:float, rng = None)->tuple[BaseDictLoader,BaseDictLoader]:

        X_train, X_test, y_train, y_test = ms.train_test_split(
            self.image_paths,
            self.label_paths,
            test_size = validation_percentage,
            random_state = rng
        )

        train = CamVidImgDictLoader(X_train,y_train,self.label_mapping,self.name_mapping)
        val = CamVidImgDictLoader(X_test,y_test,self.label_mapping,self.name_mapping)

        return train,val

    @staticmethod
    def create_from_dataset_root(
        camvid_root: os.PathLike,
        mapping_filename = 'default.txt',
        label_colors_filename = 'label_colors.txt',
        name_mapping = None,
    ):
        label_mapping = _read_label_map(camvid_root / str(label_colors_filename))
        pairs_paths = read_camvid_pairs(camvid_root / str(mapping_filename))
        img_paths,label_paths =  list(zip(*pairs_paths))

        if "img" in label_mapping:
            raise Exception(f"Keyword 'img' present in labels of dataset {camvid_root}")
            
        return CamVidImgDictLoader(
            [camvid_root/p for p in img_paths],
            [camvid_root/p for p in label_paths],
            label_mapping,
            name_mapping
        )

class CVRImgDictLoader(BaseDictLoader):
    def __init__(
        self,
        image_dirs: Sequence[os.PathLike],
        image_filenames:  Sequence[os.PathLike] = None,
        patch_shape: Optional[tuple[int,int]] = None,
        assume_grayscale = True,
        name_mapping:dict[str,str] = None,
        max_val = None,
    ):
        self.image_dirs = list(map(Path, image_dirs))
        self.image_filenames = image_filenames
        self.patch_shape = patch_shape
        self.assume_grayscale=assume_grayscale
        self.name_mapping = name_mapping
        self.max_val = max_val

    def __len__(self) -> int:
        return len(self.image_dirs)

    def _sample_origin(self, H: int, W: int, ph: int, pw: int, rng: np.random.Generator) -> tuple[int, int]:
        y0 = 0 if H <= ph else int(rng.integers(0, H - ph + 1))
        x0 = 0 if W <= pw else int(rng.integers(0, W - pw + 1))
        return y0, x0

    def __getitem__(self, idx: int):
        rng = _rng_from_worker()
        oio_dir = self.image_dirs[idx]

        image_filenames = self.image_filenames if self.image_filenames is not None else [p for p in oio_dir.glob('*') if is_img(p) ]
        if self.patch_shape is None:
            # just read img any way possible
            imgs = {}
            for img_filename in image_filenames:
                img_path = oio_dir/img_filename
                
                img = np.float32(imageio.imread(img_path))
                if self.assume_grayscale and len(img.shape) >= 3:
                    img = img[:,:,0]
                if self.max_val:
                    img = img/self.max_val
                    
                imgs[img_path.stem] = img
            return _remap_names(imgs,self.name_mapping)
        else:
            y0,x0 = None,None
            ph, pw = self.patch_shape
            patches = {}
            for img_filename in image_filenames:
                img_path = oio_dir/img_filename
    
                if y0 is None or x0 is None:
                    H, W = tiff_get_page_shape(img_path)
                    y0, x0 = self._sample_origin(H, W, ph, pw, rng)

                if img_path.suffix in ['.tif','.tiff']:
                    # try reading just the patch without loading full image
                    patch = tiff_read_region(img_path, y0, x0, ph, pw)
                else:
                    img = np.float32(imageio.imread(img_path))
                    patch  = img[y0:y0+ph,x0:x0+pw]
    
                # assume grayscale
                if self.assume_grayscale and len(patch.shape) >=3:
                    patch = patch[:,:,0]
                if self.max_val:
                    patch = patch/self.max_val
                    
                patches[img_path.stem] = patch
                
            return _remap_names(patches,self.name_mapping)

    def split_train_val(self,validation_percentage:float, rng = None)->tuple[BaseDictLoader,BaseDictLoader]:

        tr,tt = ms.train_test_split(
            self.image_dirs,
            test_size = validation_percentage,
            random_state = rng
        )

        train = CVRImgDictLoader(tr,self.image_filenames,self.patch_shape, self.assume_grayscale,self.name_mapping,self.max_val)
        val = CVRImgDictLoader(tt,self.image_filenames,self.patch_shape,self.assume_grayscale,self.name_mapping,self.max_val)

        return train,val

def tiff_get_page_shape(path: str) -> tuple[int, int]:
    with tiff.TiffFile(path) as tf:
        page = tf.pages[0]
        h, w = page.shape[:2]
    return int(h), int(w)

def _rng_from_worker() -> np.random.Generator:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(info.seed)


def is_img(p):
    return p.suffix in IMAGE_EXTS

@lru_cache(maxsize=256)
def _open_as_zarr_array(path: str):
    """
    Per-process LRU cache. Each DataLoader worker is its own process, so this cache
    is local to the worker and safe.

    Uses tifffile's aszarr to enable chunked region reads for tiled/chunked TIFFs.
    Requires 'zarr' installed.
    """
    
    store = tiff.imread(path, aszarr=True)
    # zarr.open works for both v2 and v3 stores; mode='r' ensures no writes.
    return zarr.open(store, mode="r")


def tiff_read_region(path: str, y0: int, x0: int, h: int, w: int) -> npt.NDArray:
    """
    Read only the needed region, best-effort.
    - Fast path: TIFF -> Zarr -> slice (reads only required chunks/tiles).
    - Fallback: loads full image then slices (works but slower).
    """
    try:
        arr = _open_as_zarr_array(path)
        return np.asarray(arr[y0:y0 + h, x0:x0 + w])
    except Exception:
        # Fallback (slower): full decode
        full = tiff.imread(path)
        return full[y0:y0 + h, x0:x0 + w]



def read_camvid_pairs(path: os.PathLike) -> list[tuple[os.PathLike, os.PathLike]]:
    """
    Read CamVid file pairs from a text file.

    Each line in the file is expected to contain two paths separated by ' ',
    referring to an image and its corresponding label file. This format is used
    in datasets like CamVid.

    Example line in the file:
        default/0001TP_006690.png /defaultannot/0001TP_006690_L.png

    Parameters
    ----------
    path : os.PathLike
        Path to the `.txt` file containing the dataset file pairs.

    Returns
    -------
    list of tuple of os.PathLike
        A list of (image_path, label_path) pairs with leading slashes removed.
    """
    with open(path) as f:
        sep = " "  # additinal stripping of '/' to make paths relative
        parts = (line.strip().split(sep) for line in f.readlines())
        return [[Path(pp.lstrip("/")) for pp in p] for p in parts if len(p) == 2]


def _read_label_map(label_path):
    with open(label_path) as f:
        l_split = (line.split() for line in f.readlines())
        return {lbl_id: [int(r), int(g), int(b)] for r, g, b, lbl_id in l_split}


def _get_color_as_bin(lbl, color):
    bool_mask = lbl == np.array(color)[None, None]
    color_match = np.all(bool_mask, axis=2)
    return np.float32(color_match)


def _colors_to_masks(lbl, labels_dict):
    return {
        label_id: _get_color_as_bin(lbl, col) for label_id, col in labels_dict.items()
    }


def _remap_names(data_dict,name_mapping):
    if name_mapping is None:
        return data_dict
        
    mapped = {}
    for k,v in data_dict.items():
        
        new_name = name_mapping.get(k)
        if new_name is not None:
            mapped[new_name] = v
        else:
            mapped[k] = v
    return mapped


def _imread_and_normalize(img_path,assume_grayscale):
    img = imageio.imread(img_path)