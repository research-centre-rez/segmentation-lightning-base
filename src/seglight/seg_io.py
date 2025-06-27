import logging
import os

import imageio
import numpy as np

import seglight.image_utils as iu
from seglight.domain import Image

logger = logging.getLogger("seglight.io")


def imread_as_float(
    image_path: os.PathLike,
) -> Image:
    image = imageio.imread(image_path)
    match image.shape:
        case (_, _):
            image = image.astype(np.float32)
        case (_, _, 1):
            image = image[:, :, 0].astype(np.float32)
        case (_, _, 2):
            # gray scale with alpha
            image = image[:, :, 0].astype(np.float32)
        case _:
            image = rgb2gray(image)

    image = iu.normalize_image(image)
    return image.astype(np.float32)


def rgb2gray(rgb):
    # Values taken from XYZ
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])


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
