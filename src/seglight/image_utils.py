import warnings

import numpy as np
from torch.functional import F

from seglight.domain import Image


def tile_image_with_overlap(
    img:Image,
    tile_size:int,
    overlap:int
)-> tuple(list[Image],list[tuple[int,int]], int,tuple[int,int]):
    tiles = []
    xy = []
    h, w = img.shape[:2]
    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            tile = img[y1:y2, x1:x2]
            tiles.append(tile)
            xy.append((y1, x1))
    return tiles,xy,img.shape

def blend_tiles(tiles, xy, image_shape) -> Image:
    h, w = image_shape[:2]
    c = tiles[0].shape[2] if tiles[0].ndim == 3 else 1

    result = np.zeros((h,w,c), dtype=np.float32)
    weight = np.zeros((h,w,c), dtype=np.float32)

    for (y, x), tile in zip(xy,tiles, strict=False):
        hh, ww = tile.shape[:2]
        result[y:y+hh, x:x+ww] += tile.reshape(hh, ww, -1)
        weight[y:y+hh, x:x+ww] += 1

    result /= np.maximum(weight, 1)
    return result.squeeze()

def normalize_image(image: Image) -> Image:
    max_value = image.max()
    min_value = image.min()
    if max_value == min_value:
        warnings.warn("Image min == Image max. Doing nothing.")
        return image
    return (image - min_value) / (max_value - min_value)


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0: # noqa SIM108
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0: # noqa SIM108
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x
