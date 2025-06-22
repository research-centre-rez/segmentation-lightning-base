from typing import Dict, List, Tuple, TypeAlias

import albumentations as A
import lightning as L
import numpy as np
import numpy.typing as npt
import sklearn.model_selection as ms
from torch.utils.data import DataLoader, Dataset

import seglight.image_utils as iu
import seglight.seg_io as sio
from seglight.domain import (
    AugTransform,
    AugumentationsProtocol,
    ChannelFirstImage,
    Image,
    SegmentationPairsLoaderProtocol,
)

CV2_INTER_CUBIC = 2


class CVRFolderedDSFormat:
    def __init__(self, data_path, test_txt_path=None):
        self.data_path = data_path
        if test_txt_path is None:
            self.test_txt_path = data_path / "test.txt"
        else:
            self.test_txt_path = test_txt_path

    def load_dir_dict(self, data_paths):
        data = {}
        for key, dir_path in data_paths.items():
            data[key] = {p.stem: sio.imread_as_float(p) for p in dir_path.glob("*")}
        return data

    def read_train_test_paths(self):
        all_data = {p.name: p for p in self.data_path.glob("*") if p.is_dir()}
        with open(self.test_txt_path) as f:
            test_names = {l.strip() for l in f.readlines()}

        train_paths = {}
        test_paths = {}
        for k, p in all_data.items():
            if k in test_names:
                test_paths[k] = p
            else:
                train_paths[k] = p

        return train_paths, test_paths


class AugumentedDataset(Dataset):
    def __init__(
        self,
        images: List[Image],
        labels: List[Image],
        transform: AugTransform | None = None,
    ):
        if len(images) != len(labels):
            raise Exception(
                f"Number of images and labels doesn't match {len(images)=}!={len(labels)=}"
            )

        self.images = [np.float32(img) for img in images]
        self.labels = [np.float32(label) for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def _transform(self, image, label) -> Tuple[npt.NDArray, npt.NDArray]:
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            tr_image = transformed["image"]
            tr_label = transformed["mask"]
            return tr_image, tr_label
        else:
            return image, label

    def __getitem__(self, idx) -> Dict[str, ChannelFirstImage]:
        image = self.images[idx]
        label = self.labels[idx]
        image_aug, y = self._transform(image, label)

        x = image_aug[None]
        if len(y.shape) > 2:
            # e.g. channel first
            y = np.rollaxis(y, -1)

        return x, y


class _DummyAug:
    @property
    def val_augumentation_fn(self):
        return None

    @property
    def train_augumentation_fn(self):
        return None


class TrainTestDataModule(L.LightningDataModule):
    def __init__(
        self,
        seg_pairs_loader: SegmentationPairsLoaderProtocol,
        augumentations: AugumentationsProtocol | None = None,
        batch_size=32,
        test_batch_size=1,
        val_size=0.25,
    ):
        super().__init__()
        self.seg_pairs_loader = seg_pairs_loader

        if augumentations:
            self.aug = augumentations
        else:
            self.aug = _DummyAug()

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_size = val_size

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            imgs, labels = self.seg_pairs_loader.load("train")

            img_train, img_val, label_train, label_val = ms.train_test_split(
                imgs, labels, test_size=self.val_size
            )

            dataset_train = AugumentedDataset(
                img_train, label_train, self.aug.train_augumentation_fn
            )

            self.train_dl = DataLoader(
                dataset_train,
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=True,
            )

            dataset_val = AugumentedDataset(
                img_val, label_val, self.aug.val_augumentation_fn
            )

            self.val_dl = DataLoader(
                dataset_val,
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=False,
            )

        if stage is None or stage == "test":
            test_imgs, test_labels = self.seg_pairs_loader.load("test")
            test_ds = AugumentedDataset(test_imgs, test_labels)

            self.test_dl = DataLoader(
                test_ds,
                batch_size=self.test_batch_size,
                shuffle=False,
            )

        if stage is None or stage == "predict":
            pred_imgs, pred_labels = self.seg_pairs_loader.load("predict")

            pred_ds = AugumentedDataset(pred_imgs, pred_labels)

            self.pred_dl = DataLoader(
                pred_ds,
                batch_size=self.test_batch_size,
                shuffle=False,
            )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def predict_dataloader(self):
        return self.pred_dl
