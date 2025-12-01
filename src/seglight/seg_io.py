import logging
import os

import imageio
import numpy as np

import seglight.image_utils as iu
from seglight.domain import Image

logger = logging.getLogger("seglight.io")


def imread_as_float(
    image_path: os.PathLike,
    keep_channels=False,
) -> Image:
    image = imread_raw(image_path)
    match image.shape:
        case (_, _):
            image = image.astype(np.float32)
        case (_, _, 1):
            image = image[:, :, 0].astype(np.float32)
        case (_, _, 2):
            # gray scale with alpha
            image = image[:, :, 0].astype(np.float32)
        case _:
            if not keep_channels:
                image = iu.rgb2gray(image)

    image = iu.normalize_image(image)
    return image.astype(np.float32)


def imread_raw(image_path: os.PathLike):
    return imageio.imread(image_path)


def imwrite_1ch(
    image_path: os.PathLike,
    img_float_single_channel: Image,
):
    img_uint8 = np.uint8(img_float_single_channel * 255)
    img_uint8_3ch = np.dstack([img_uint8] * 3)
    imageio.imwrite(image_path, img_uint8_3ch)


def imwrite_3ch(
    image_path: os.PathLike,
    img_uint8_three_channel: Image,
):
    imageio.imwrite(image_path, img_uint8_three_channel)
