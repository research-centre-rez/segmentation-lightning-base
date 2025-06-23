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


class CVRFolderedDSFormat:
    """
    A dataset loader for image samples organized into directories with
    test/train split.

    Each sample is stored in its own subdirectory within `data_path`, and
    contains images such as `img.png`, `label.png`, `oxides.png`, etc. A
    `test.txt` file specifies which samples should be used as test data.

    Parameters
    ----------
    data_path : Path
        Path to the directory containing the dataset subfolders.
    test_txt_path : Path, optional
        Path to a text file listing test sample names (one per line).
        If not provided, defaults to `data_path / "test.txt"`.

    Attributes
    ----------
    data_path : Path
        Path to the dataset root directory.
    test_txt_path : Path
        Path to the test split definition file.
    """
    def __init__(self, data_path, test_txt_path=None):
        self.data_path = data_path
        if test_txt_path is None:
            self.test_txt_path = data_path / "test.txt"
        else:
            self.test_txt_path = test_txt_path

    def load_dir_dict(self, data_paths):
        """
        Load images from multiple sample directories into a dictionary.

        Each directory should contain image files named by type, e.g. `img.png`,
        `label.png`, etc.



        Parameters
        ----------
        data_paths : dict of str to Path
            Dictionary mapping sample names to their corresponding directory
            paths.

        Returns
        -------
        dict of str to dict of str to ndarray
            A nested dictionary where the outer key is the sample name and the
            inner dictionary maps image type (from filename stem e.g.
            `label.png` -> `label`) to the corresponding image array.
        """
        data = {}
        for key, dir_path in data_paths.items():
            data[key] = {p.stem: sio.imread_as_float(p) for p in dir_path.glob("*")}
        return data

    def read_train_test_paths(self):
        """
        Split the dataset into training and testing subsets.

        This method reads subdirectories in `data_path` and compares their names to
        entries in `test_txt_path` to determine their assignment.

        Returns
        -------
        train_paths : dict of str to Path
            Dictionary mapping training sample names to their directory paths.
        test_paths : dict of str to Path
            Dictionary mapping test sample names to their directory paths.
        """
        all_data = {p.name: p for p in self.data_path.glob("*") if p.is_dir()}
        with open(self.test_txt_path) as f:
            test_names = {line.strip() for line in f.readlines()}

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
                "{len(images)=}!={len(labels)=}"
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

    def __getitem__(self, idx) -> dict[str, ChannelFirstImage]:
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
    def __init__( # PLR0913
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
            self.test_dl = self._read_data_pairs(
                "test",
                self.test_batch_size
            )

        if stage is None or stage == "predict":
            self.pred_dl = self._read_data_pairs(
                "predict",
                self.test_batch_size
            )


    def _read_data_pairs(self, set_name,batch_size):
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
