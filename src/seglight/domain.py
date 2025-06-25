from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

ChannelFirstImage: TypeAlias = npt.NDArray[np.float32]
Image: TypeAlias = npt.NDArray[np.float32]
BinaryImage: TypeAlias = npt.NDArray[np.uint8]
ImageDirDict: TypeAlias = dict[str, Image]
AugOutput: TypeAlias = dict[str, Image]


class SegmentationPairsLoaderProtocol(Protocol):
    def load(self, set_type: str) -> tuple[list[Image], list[Image]]:
        # accepts train test predict
        ...


class AugTransform(Protocol):
    def __call__(self, image: Image, mask: Image) -> AugOutput: ...


class AugumentationsProtocol(Protocol):
    @property
    def train_augumentaions_fn(self) -> AugTransform: ...
    def val_augumentaions_fn(self) -> AugTransform: ...
