from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

import yaml
from tqdm.cli import tqdm

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def load_data_yaml(dataset_root: Path) -> dict:
    yaml_path = dataset_root / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Could not find data.yaml in {dataset_root}"
        )

    with yaml_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_root(
    input_root: Path,
    data: dict,
) -> Path:
    """
    Resolve the effective dataset root from data.yaml.

    If `path:` is absent, the directory containing data.yaml is used.
    """
    path_value = data.get("path")

    if path_value is None:
        return input_root.resolve()

    path = Path(path_value)

    if not path.is_absolute():
        path = input_root / path

    return path.resolve()


def resolve_train_dir(
    input_root: Path,
    data: dict,
) -> Path:
    """
    Resolve the train image directory defined by data.yaml.

    This splitter intentionally expects the common case:

        train: images/train
    """
    train_value = data.get("train")

    if train_value is None:
        raise ValueError(
            "data.yaml does not contain a 'train' entry."
        )

    if isinstance(train_value, list):
        raise ValueError(
            "This splitter expects 'train' to point to one directory, "
            "not a list of directories."
        )

    dataset_root = resolve_dataset_root(
        input_root,
        data,
    )

    train_dir = Path(train_value)

    if not train_dir.is_absolute():
        train_dir = dataset_root / train_dir

    train_dir = train_dir.resolve()

    if not train_dir.is_dir():
        raise NotADirectoryError(
            f"Training image directory does not exist: {train_dir}"
        )

    return train_dir


def find_label_path(
    image_path: Path,
) -> Path:
    """
    Convert:

        dataset/images/train/foo.png

    into:

        dataset/labels/train/foo.txt
    """
    parts = list(image_path.parts)

    image_indices = [
        i
        for i, part in enumerate(parts)
        if part == "images"
    ]

    if not image_indices:
        raise ValueError(
            "Cannot determine label path because image path does "
            f"not contain an 'images' directory: {image_path}"
        )

    index = image_indices[-1]

    label_parts = (
        parts[:index]
        + ["labels"]
        + parts[index + 1 :]
    )

    return Path(*label_parts).with_suffix(".txt")


def get_images(train_dir: Path) -> list[Path]:
    images = sorted(
        path
        for path in train_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not images:
        raise ValueError(
            f"No images found in {train_dir}"
        )

    return images


def copy_sample(
    image_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
) -> None:
    """
    Copy one image and its corresponding YOLO label.

    A missing source label is treated as a negative/background-only image
    and results in an empty output label.
    """
    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_image = (
        output_image_dir
        / image_path.name
    )

    destination_label = (
        output_label_dir
        / f"{image_path.stem}.txt"
    )

    shutil.copy2(
        image_path,
        destination_image,
    )

    source_label = find_label_path(
        image_path
    )

    if source_label.exists():
        shutil.copy2(
            source_label,
            destination_label,
        )
    else:
        destination_label.write_text(
            "",
            encoding="utf-8",
        )


def write_data_yaml(
    output_root: Path,
    input_data: dict,
) -> None:
    """
    Write the output Ultralytics data.yaml while retaining the class names.
    """
    if "names" not in input_data:
        raise ValueError(
            "Input data.yaml does not contain 'names'."
        )

    output_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": input_data["names"],
    }

    # Preserve nc if the original file explicitly had it.
    if "nc" in input_data:
        output_data["nc"] = input_data["nc"]

    with (
        output_root / "data.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            output_data,
            f,
            sort_keys=False,
        )


def split_dataset(
    input_path: Path,
    output_path: Path,
    val_size: float,
    seed: int,
) -> None:
    if not 0.0 < val_size < 1.0:
        raise ValueError(
            f"--val-size must be between 0 and 1, got {val_size}"
        )

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if input_path == output_path:
        raise ValueError(
            "Input and output directories must be different."
        )

    data = load_data_yaml(
        input_path
    )

    train_dir = resolve_train_dir(
        input_path,
        data,
    )

    images = get_images(
        train_dir
    )

    if len(images) < 2:
        raise ValueError(
            "At least two images are required to create "
            "train and validation sets."
        )

    rng = random.Random(seed)

    images = images.copy()
    rng.shuffle(images)

    # Equivalent to the usual interpretation of a fractional
    # validation/test size: ensure at least the requested proportion.
    n_val = math.ceil(
        len(images) * val_size
    )

    # Ensure both splits remain non-empty.
    n_val = max(
        1,
        min(
            n_val,
            len(images) - 1,
        ),
    )

    val_images = images[:n_val]
    train_images = images[n_val:]

    if output_path.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_path}"
        )

    output_path.mkdir(
        parents=True
    )

    for image_path in tqdm(train_images,desc='train'):
        copy_sample(
            image_path,
            output_path / "images" / "train",
            output_path / "labels" / "train",
        )

    for image_path in tqdm(val_images,desc='val'):
        copy_sample(
            image_path,
            output_path / "images" / "val",
            output_path / "labels" / "val",
        )

    write_data_yaml(
        output_root=output_path,
        input_data=data,
    )

    print(
        f"Total: {len(images)}\n"
        f"Train: {len(train_images)} "
        f"({len(train_images) / len(images):.1%})\n"
        f"Val:   {len(val_images)} "
        f"({len(val_images) / len(images):.1%})\n"
        f"Output: {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly split an Ultralytics segmentation "
            "training dataset into train and validation sets."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help=(
            "Input Ultralytics dataset directory containing data.yaml."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        required=True,
        help=(
            "Fraction of samples assigned to validation, "
            "e.g. 0.20."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    args = parser.parse_args()

    split_dataset(
        input_path=args.input_path,
        output_path=args.output,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
