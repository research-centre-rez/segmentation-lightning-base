import os
import shutil
import tempfile
import warnings
from pathlib import Path

import lightning as L
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader, Dataset

import seglight.seg_io as sio
from seglight.domain import (
    AugTransform,
    AugumentationsProtocol,
    ChannelFirstImage,
    Image,
    SegmentationPairsLoaderProtocol,
)

CV2_INTER_CUBIC = 2


class CamVidDSFormat:
    """
    Class implementing handling of CamVid files
    See: https://docs.cvat.ai/docs/manual/advanced/formats/format-camvid/
    """

    def __init__(self, dataset_root: os.PathLike, txt_name="default.txt"):
        self.dataset_root = Path(dataset_root)
        self.txt_name = txt_name

    def load_train(self):
        dr = self.dataset_root

        label_dict = _read_label_map(dr / "label_colors.txt")
        pairs_paths = read_camvid_pairs(dr / self.txt_name)

        data = {}
        for img_p, lbl_p in pairs_paths:
            name = img_p.stem
            img = sio.imread_as_float(dr / str(img_p))
            lbl_ch = sio.imread_raw(dr / str(lbl_p))

            lbls = _colors_to_masks(lbl_ch, label_dict)

            data[name] = {"img": img} | lbls
        return data

    def load_test(self):
        warnings.warn("No test in Camvid DS Format")
        return {}

    @staticmethod
    def dump(
        destination: os.PathLike,
        dataset: dict[str, dict[str, Image]],
        imgs_dir_name="default",
        labels_dir_name="defaultannot",
        label_dataset_key="label",
        img_dataset_key="img",
        rgb_label_color=(255, 255, 100),
    ):
        if Path(destination).exists():
            raise Exception(f"Destination {destination} exists!")

        tmp_dir = Path(tempfile.mkdtemp())

        img_paths = []
        label_paths = []
        for sample_name, data in dataset.items():
            img = data[img_dataset_key]
            img_path = tmp_dir / imgs_dir_name / f"{sample_name}_img.png"
            img_paths.append(img_path)
            img_path.parent.mkdir(exist_ok=True, parents=True)
            sio.imwrite_1ch(img_path, img)

            label = data[label_dataset_key]
            label_path = tmp_dir / labels_dir_name / f"{sample_name}_label.png"
            label_paths.append(label_path)
            label_path.parent.mkdir(exist_ok=True, parents=True)
            label_rgb = _colorize_gs_image(label, rgb_label_color)
            sio.imwrite_3ch(label_path, label_rgb)

        lbl_colors_path = tmp_dir / "label_colors.txt"
        r, g, b = [int(c) for c in rgb_label_color]
        with open(lbl_colors_path, "w") as f:
            f.write(f"{r} {g} {b} foreground\n")

        with open(tmp_dir / "default.txt", "w") as f:
            for img_p, lib_p in zip(img_paths, label_paths, strict=False):
                line = f"/{'/'.join(img_p.parts[-2:])} /{'/'.join(lib_p.parts[-2:])}\n"
                f.write(line)

        shutil.move(tmp_dir, destination)


class CVRFolderedDSFormat:
    def __init__(
        self, data_path: os.PathLike, test_txt_path: str | None = None, no_test=False
    ):
        """
        A dataset format handler for CVR-style foldered datasets.

        This class manages dataset paths and test file configuration
        based on provided arguments, allowing optional disabling of test data.

        Parameters
        ----------
        data_path : pathlib.Path or str
            Path to the root directory of the dataset.

        test_txt_path : pathlib.Path or str, optional
            Path to the test set file. If None and `no_test` is False,
            defaults to `data_path / "test.txt"`.

        no_test : bool, optional
            If True, disables the use of test file entirely by setting
            `self.test_txt_path` to None. If False and `test_txt_path` is None,
            the default path is used.
        """
        self.data_path = data_path

        if no_test:
            self.test_txt_path = None
        elif test_txt_path is None:
            self.test_txt_path = Path(data_path) / "test.txt"
        else:
            self.test_txt_path = Path(test_txt_path)

    def load_train(self) -> dict[str, dict[str, Image]]:
        """
        Load train images from multiple sample directories into a dictionary.

        Each directory should contain image files named by type, e.g. `img.png`,
        `label.png`, etc.


        Returns
        -------
        dict of str to dict of str to Image
            A nested dictionary where the outer key is the sample name and the
            inner dictionary maps image type (from filename stem e.g.
            `label.png` -> `label`) to the corresponding image array.
        """
        train_paths, _ = self._read_train_test_paths()
        return self._load_dir(train_paths)

    def load_test(self) -> dict[str, dict[str, Image]]:
        """
        Load train images from multiple sample directories into a dictionary.

        Each directory should contain image files named by type, e.g. `img.png`,
        `label.png`, etc.


        Returns
        -------
        dict of str to dict of str to Image
            A nested dictionary where the outer key is the sample name and the
            inner dictionary maps image type (from filename stem e.g.
            `label.png` -> `label`) to the corresponding image array.
        """
        _, test_paths = self._read_train_test_paths()
        return self._load_dir(test_paths)

    def _load_dir(self, data_paths):
        data = {}
        for key, dir_path in data_paths.items():
            data[key] = {
                p.stem: sio.imread_as_float(p)
                for p in dir_path.glob("*")
                if not p.is_dir()
            }
        return data

    def _read_train_test_paths(self):
        all_data = {
            p.name: p
            for p in self.data_path.glob("*")
            if p.is_dir() and p.name[0] != "."  # avoid dotfolder (.git)
        }

        if self.test_txt_path is not None:
            with open(self.test_txt_path) as f:
                test_names = {line.strip() for line in f.readlines()}
        else:
            test_names = {}

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
        images: list[Image],
        labels: list[Image],
        transform: AugTransform | None = None,
    ):
        if len(images) != len(labels):
            raise Exception(
                "Number of images and labels doesn't match "
                f"{len(images)=}!={len(labels)=}"
            )

        self.images = [np.float32(img) for img in images]
        self.labels = [np.float32(label) for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def _transform(self, image, label) -> tuple[Image, Image]:
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            tr_image = transformed["image"]
            tr_label = transformed["mask"]
            return tr_image, tr_label

        return image, label

    def __getitem__(self, idx) -> tuple[ChannelFirstImage, ChannelFirstImage]:
        image = self.images[idx]
        label = self.labels[idx]
        image_aug, y = self._transform(image, label)

        if len(image_aug.shape) == 2:
            # (H,W) -> (1,H,W)
            x = image_aug[None]
        else:
            # (H,W,C) -> (C,H,W)
            x = np.rollaxis(image_aug, -1)

        if len(y.shape) > 2:
            # (H,W,C) -> (C,H,W)
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
    def __init__(  # PLR0913
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

    def setup(self, stage: str):
        """
        Set up the environment based on the specified stage.

        This method is typically used in PyTorch Lightning modules to set up
        data processing logic according to the current stage of execution
        (e.g., 'fit', 'validate', 'test', or 'predict').

        See: https://lightning.ai/docs/pytorch/stable/data/datamodule.html

        Parameters
        ----------
        stage : str
            The stage for which the setup is being called. Common values are
            'fit', 'validate', 'test', or 'predict'.

        """
        if stage in {"fit", "validate"}:
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

        elif stage == "test":
            self.test_dl = self._read_data_pairs("test", self.test_batch_size)
        elif stage == "predict":
            self.pred_dl = self._read_data_pairs("predict", self.test_batch_size)
        else:
            warnings.warn(
                f"Parameter {stage=} was not recognized."
                "Use 'fit', 'validate', 'predict' or 'test'"
            )

    def _read_data_pairs(self, set_name, batch_size):
        imgs, labels = self.seg_pairs_loader.load(set_name)
        ds = AugumentedDataset(imgs, labels)
        return DataLoader(
            ds,
            batch_size=batch_size,
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


def read_camvid_pairs(path: os.PathLike):
    with open(path) as f:
        sep = " /"
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


def _colorize_gs_image(image: Image, color_rgb: list[int]):
    color = np.array(color_rgb)
    img_rgb = np.dstack([image] * 3) * color
    return np.uint8(img_rgb)
