"""
python segmentation_converter.py camvid \
    --root path/to/dataset \
    --class grids=1 \
    --class rods=2 \
    --output path/to/output

python segmentation_converter.py typed \
    --root path/to/dataset \
    --split train=. \
    --class-mask grids=grids.png \
    --class-mask rods=rods.png \
    --output path/to/output

python segmentation_converter.py merge \
    --data dataset_a/data.yaml dataset_b/data.yaml \
    --output merged_dataset
"""
from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm.cli import tqdm

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================================
# Common utilities
# ============================================================================


def parse_key_value(value: str) -> tuple[str, str]:
    """
    Parse a CLI argument of the form KEY=VALUE.
    """
    try:
        key, value = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE, got {value!r}"
        ) from exc

    return key, value


def load_single_channel_image(path: Path) -> np.ndarray:
    """
    Load a single-channel image/mask.
    """
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    if image.ndim == 3:
        if image.shape[2] in [0,3,4]:
            image = image[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel mask, "
                f"got shape {image.shape}: {path}"
            )

    return image


def load_binary_mask(path: Path) -> np.ndarray:
    """
    Load a binary class mask.

    Zero:
        background

    Non-zero:
        foreground
    """
    return load_single_channel_image(path) != 0


def binary_mask_to_polygons(
    binary_mask: np.ndarray,
    min_area: int = 1,
) -> list[np.ndarray]:
    """
    Convert a binary semantic mask to instance polygons.

    Each connected component is treated as one instance.

    Returns
    -------
    list[np.ndarray]
        Polygons with shape (N, 2) in pixel coordinates.
    """
    binary_mask = binary_mask.astype(np.uint8)

    n_components, component_map, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    polygons: list[np.ndarray] = []

    # Component 0 is the background.
    for component_id in range(1, n_components):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        component_mask = (
            component_map == component_id
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            polygon = contour.reshape(-1, 2)

            # Ultralytics segmentation requires at least
            # three polygon points.
            if len(polygon) < 3:
                continue

            polygons.append(polygon)

    return polygons


def polygon_to_yolo_line(
    polygon: np.ndarray,
    class_id: int,
    width: int,
    height: int,
) -> str:
    """
    Convert an Nx2 polygon from pixel coordinates into one
    Ultralytics segmentation label row:

        class_id x1 y1 x2 y2 ... xn yn

    Coordinates are normalized to [0, 1].
    """
    polygon = polygon.astype(np.float64).copy()

    polygon[:, 0] /= width
    polygon[:, 1] /= height

    polygon = np.clip(
        polygon,
        0.0,
        1.0,
    )

    coordinates = " ".join(
        f"{value:.6f}"
        for value in polygon.reshape(-1)
    )

    return f"{class_id} {coordinates}"


def write_data_yaml(
    output_root: Path,
    classes: Iterable[str],
    splits: Iterable[str],
) -> None:
    """
    Write an Ultralytics data.yaml.
    """
    output_root = output_root.resolve()
    splits = set(splits)

    data: dict = {
        "path": str(output_root),
    }

    for split in ("train", "val", "test"):
        if split in splits:
            data[split] = f"images/{split}"

    data["names"] = {
        i: name
        for i, name in enumerate(classes)
    }

    yaml_path = output_root / "data.yaml"

    with yaml_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
        )


# ============================================================================
# CamVid source format
# ============================================================================


def is_camvid_pairs_file(
    path: Path,
    source_root: Path,
) -> bool:
    """
    Check whether a text file looks like a CamVid image/mask manifest.

    Expected format:

        path/to/image.png path/to/mask.png
        path/to/image.png path/to/mask.png

    Blank lines and comments are ignored.
    """
    found_pair = False

    try:
        with path.open(
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) != 2:
                    return False

                image_path = Path(parts[0])
                mask_path = Path(parts[1])

                if not image_path.is_absolute():
                    image_path = (
                        source_root / image_path
                    )

                if not mask_path.is_absolute():
                    mask_path = (
                        source_root / mask_path
                    )

                if not image_path.exists():
                    return False

                if not mask_path.exists():
                    return False

                found_pair = True

    except (OSError, UnicodeDecodeError):
        return False

    return found_pair


def read_camvid_pairs(
    pairs_file: Path,
    source_root: Path,
) -> list[tuple[Path, Path]]:
    """
    Read a CamVid-style image/mask pair file.

    Example:

        images/a.png masks/a.png
        images/b.png masks/b.png

    Relative paths are resolved against source_root.
    """
    pairs: list[tuple[Path, Path]] = []

    with pairs_file.open(
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"{pairs_file}:{line_number}: "
                    f"expected '<image> <mask>', "
                    f"got {line!r}"
                )

            image_path = Path(parts[0])
            mask_path = Path(parts[1])

            if not image_path.is_absolute():
                image_path = (
                    source_root / image_path
                )

            if not mask_path.is_absolute():
                mask_path = (
                    source_root / mask_path
                )

            image_path = image_path.resolve()
            mask_path = mask_path.resolve()

            if not image_path.exists():
                raise FileNotFoundError(
                    image_path
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    mask_path
                )

            pairs.append(
                (
                    image_path,
                    mask_path,
                )
            )

    return pairs


def discover_camvid_splits(
    source_root: Path,
    explicit_splits: list[tuple[str, str]] | None = None,
    pairs_file: Path | None = None,
) -> dict[str, Path]:
    """
    Discover which CamVid manifest(s) to use.

    Priority
    --------
    1. Explicit --split arguments.
    2. Explicit --pairs argument.
    3. train.txt / val.txt / test.txt in the dataset root.
    4. Exactly one valid root-level *.txt CamVid manifest.

    A single unsplit manifest becomes the `train` split.
    """
    source_root = source_root.resolve()

    if explicit_splits and pairs_file is not None:
        raise ValueError(
            "--split and --pairs cannot be used together."
        )

    # ------------------------------------------------------------------
    # Explicit split files.
    # ------------------------------------------------------------------

    if explicit_splits:
        result: dict[str, Path] = {}

        for split, filename in explicit_splits:
            path = Path(filename)

            if not path.is_absolute():
                path = source_root / path

            path = path.resolve()

            if not path.exists():
                raise FileNotFoundError(path)

            if not is_camvid_pairs_file(
                path,
                source_root,
            ):
                raise ValueError(
                    f"File does not look like a valid CamVid "
                    f"image/mask manifest: {path}"
                )

            result[split] = path

        return result

    # ------------------------------------------------------------------
    # Explicit single pair manifest.
    # ------------------------------------------------------------------

    if pairs_file is not None:
        path = Path(pairs_file)

        if not path.is_absolute():
            path = source_root / path

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        if not is_camvid_pairs_file(
            path,
            source_root,
        ):
            raise ValueError(
                f"File does not look like a valid CamVid "
                f"image/mask manifest: {path}"
            )

        return {
            "train": path,
        }

    # ------------------------------------------------------------------
    # Conventional split files.
    # ------------------------------------------------------------------

    conventional: dict[str, Path] = {}

    for split in ("train", "val", "test"):
        path = (
            source_root
            / f"{split}.txt"
        )

        if (
            path.exists()
            and is_camvid_pairs_file(
                path,
                source_root,
            )
        ):
            conventional[split] = (
                path.resolve()
            )

    if conventional:
        return conventional

    # ------------------------------------------------------------------
    # Look for one root-level unsplit manifest.
    # ------------------------------------------------------------------

    manifests = [
        path.resolve()
        for path in sorted(
            source_root.glob("*.txt")
        )
        if is_camvid_pairs_file(
            path,
            source_root,
        )
    ]

    if len(manifests) == 1:
        return {
            "train": manifests[0],
        }

    if not manifests:
        raise FileNotFoundError(
            "Could not find a CamVid image/mask manifest "
            f"in:\n  {source_root}\n\n"
            "Expected one of:\n"
            "  - train.txt / val.txt / test.txt\n"
            "  - one root-level image/mask .txt manifest\n"
            "  - --pairs FILE\n"
            "  - --split NAME=FILE"
        )

    raise ValueError(
        "Found multiple possible CamVid manifests but "
        "cannot determine which one to use:\n"
        + "\n".join(
            f"  {path}"
            for path in manifests
        )
        + "\n\nSpecify one explicitly with --pairs, "
        "or specify splits with --split."
    )


def convert_camvid_split(
    pairs_file: Path,
    source_root: Path,
    output_root: Path,
    split: str,
    classes: dict[str, int],
    min_area: int = 1,
) -> None:
    """
    Convert one CamVid semantic-segmentation split to
    Ultralytics instance segmentation.

    classes maps:

        output class name -> integer value in source mask

    Example:

        {
            "grids": 1,
            "rods": 2,
        }
    """
    pairs = read_camvid_pairs(
        pairs_file=pairs_file,
        source_root=source_root,
    )

    image_output_dir = (
        output_root
        / "images"
        / split
    )

    label_output_dir = (
        output_root
        / "labels"
        / split
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_ids = {
        class_name: i
        for i, class_name in enumerate(
            classes
        )
    }

    used_names: set[str] = set()

    for image_path, mask_path in pairs:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: "
                f"{image_path}"
            )

        mask = load_single_channel_image(
            mask_path
        )

        height, width = mask.shape

        if image.shape[:2] != (
            height,
            width,
        ):
            raise ValueError(
                "Image/mask size mismatch:\n"
                f"  image: {image_path} "
                f"{image.shape[:2]}\n"
                f"  mask:  {mask_path} "
                f"{mask.shape}"
            )

        output_name = image_path.name

        if output_name in used_names:
            raise ValueError(
                f"Duplicate image filename in split "
                f"{split!r}: {output_name}"
            )

        used_names.add(output_name)

        shutil.copy2(
            image_path,
            image_output_dir
            / output_name,
        )

        label_lines: list[str] = []

        for (
            class_name,
            mask_value,
        ) in classes.items():
            binary_mask = (
                mask == mask_value
            )

            polygons = (
                binary_mask_to_polygons(
                    binary_mask,
                    min_area=min_area,
                )
            )

            class_id = class_ids[
                class_name
            ]

            for polygon in polygons:
                label_lines.append(
                    polygon_to_yolo_line(
                        polygon=polygon,
                        class_id=class_id,
                        width=width,
                        height=height,
                    )
                )

        label_path = (
            label_output_dir
            / f"{image_path.stem}.txt"
        )

        label_path.write_text(
            "\n".join(label_lines)
            + (
                "\n"
                if label_lines
                else ""
            ),
            encoding="utf-8",
        )


def convert_camvid_dataset(
    source_root: Path,
    split_files: dict[str, Path],
    output_root: Path,
    classes: dict[str, int],
    min_area: int = 1,
) -> None:
    """
    Convert a CamVid dataset into Ultralytics segmentation format.
    """
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, pairs_file in (
        split_files.items()
    ):
        convert_camvid_split(
            pairs_file=pairs_file,
            source_root=source_root,
            output_root=output_root,
            split=split,
            classes=classes,
            min_area=min_area,
        )

    write_data_yaml(
        output_root=output_root,
        classes=classes.keys(),
        splits=split_files.keys(),
    )


# ============================================================================
# Typed source format
# ============================================================================


def convert_typed_split(
    split_root: Path,
    output_root: Path,
    split: str,
    image_name: str,
    class_masks: dict[str, str],
    min_area: int = 1,
) -> None:
    """
    Convert a custom typed dataset.

    Expected structure:

        split_root/
            sample_01/
                img.png
                grids.png
                rods.png

            sample_02/
                img.png
                grids.png
                rods.png

    class_masks maps:

        output class name -> filename inside each sample directory

    Example:

        {
            "grids": "grids.png",
            "rods": "rods.png",
        }

    Class masks are binary:

        zero     -> background
        non-zero -> foreground
    """
    split_root = split_root.resolve()

    if not split_root.is_dir():
        raise NotADirectoryError(
            split_root
        )

    image_output_dir = (
        output_root
        / "images"
        / split
    )

    label_output_dir = (
        output_root
        / "labels"
        / split
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_ids = {
        class_name: i
        for i, class_name in enumerate(
            class_masks
        )
    }

    sample_dirs = sorted(
        path
        for path in split_root.iterdir()
        if path.is_dir()
    )

    if not sample_dirs:
        raise ValueError(
            f"No sample directories found in "
            f"{split_root}"
        )

    for sample_dir in tqdm(sample_dirs,desc = f"split: {split_root}"):
        image_path = (
            sample_dir
            / image_name
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing for sample "
                f"{sample_dir.name}: "
                f"{image_path}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: "
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        label_lines: list[str] = []

        for (
            class_name,
            mask_filename,
        ) in class_masks.items():
            mask_path = (
                sample_dir
                / mask_filename
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask {class_name!r} missing "
                    f"for sample {sample_dir.name}: "
                    f"{mask_path}"
                )

            binary_mask = (
                load_binary_mask(
                    mask_path
                )
            )

            if binary_mask.shape != (
                height,
                width,
            ):
                raise ValueError(
                    "Image/mask size mismatch:\n"
                    f"  sample: "
                    f"{sample_dir.name}\n"
                    f"  image: "
                    f"{image.shape[:2]}\n"
                    f"  mask:  "
                    f"{binary_mask.shape} "
                    f"({mask_path.name})"
                )

            polygons = (
                binary_mask_to_polygons(
                    binary_mask,
                    min_area=min_area,
                )
            )

            class_id = (
                class_ids[
                    class_name
                ]
            )

            for polygon in polygons:
                label_lines.append(
                    polygon_to_yolo_line(
                        polygon=polygon,
                        class_id=class_id,
                        width=width,
                        height=height,
                    )
                )

        # All source images may be called img.png,
        # therefore the sample directory name is used
        # as the output filename.
        output_image_name = (
            sample_dir.name
            + image_path.suffix.lower()
        )

        shutil.copy2(
            image_path,
            image_output_dir
            / output_image_name,
        )

        label_path = (
            label_output_dir
            / f"{sample_dir.name}.txt"
        )

        label_path.write_text(
            "\n".join(label_lines)
            + (
                "\n"
                if label_lines
                else ""
            ),
            encoding="utf-8",
        )


def convert_typed_dataset(
    split_dirs: dict[str, Path],
    output_root: Path,
    image_name: str,
    class_masks: dict[str, str],
    min_area: int = 1,
) -> None:
    """
    Convert a typed dataset into Ultralytics segmentation format.
    """
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, split_root in (
        split_dirs.items()
    ):
        convert_typed_split(
            split_root=split_root,
            output_root=output_root,
            split=split,
            image_name=image_name,
            class_masks=class_masks,
            min_area=min_area,
        )

    write_data_yaml(
        output_root=output_root,
        classes=class_masks.keys(),
        splits=split_dirs.keys(),
    )


# ============================================================================
# Merge existing Ultralytics segmentation datasets
# ============================================================================


def load_names(
    data: dict,
) -> dict[int, str]:
    """
    Normalize either:

        names:
          0: rods
          1: grids

    or:

        names:
          - rods
          - grids

    into:

        {
            0: "rods",
            1: "grids",
        }
    """
    names = data["names"]

    if isinstance(names, list):
        return {
            i: name
            for i, name in enumerate(
                names
            )
        }

    return {
        int(i): name
        for i, name in names.items()
    }


def get_dataset_root(
    yaml_path: Path,
    data: dict,
) -> Path:
    """
    Resolve the root path of an Ultralytics dataset.
    """
    path_value = data.get("path")

    if path_value is None:
        return yaml_path.parent.resolve()

    path = Path(path_value)

    if not path.is_absolute():
        path = (
            yaml_path.parent
            / path
        )

    return path.resolve()


def resolve_split_images(
    yaml_path: Path,
    data: dict,
    split: str,
) -> list[Path]:
    """
    Resolve images belonging to an Ultralytics split.

    Supports:

        train: images/train

    lists:

        train:
          - images/train_a
          - images/train_b

    and text-file image lists.
    """
    spec = data.get(split)

    if not spec:
        return []

    root = get_dataset_root(
        yaml_path,
        data,
    )

    if not isinstance(spec, list):
        spec = [spec]

    images: list[Path] = []

    for entry in spec:
        path = Path(entry)

        if not path.is_absolute():
            path = root / path

        path = path.resolve()

        if path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if (
                        p.is_file()
                        and p.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                )
            )

        elif (
            path.is_file()
            and path.suffix.lower()
            == ".txt"
        ):
            with path.open(
                encoding="utf-8",
            ) as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    image_path = Path(
                        line
                    )

                    if not (
                        image_path.is_absolute()
                    ):
                        image_path = (
                            path.parent
                            / image_path
                        )

                    images.append(
                        image_path.resolve()
                    )

        elif (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            images.append(path)

        else:
            raise ValueError(
                f"Unsupported split source: "
                f"{path}"
            )

    return images


def find_label_for_image(
    image_path: Path,
) -> Path:
    """
    Map an Ultralytics image path:

        .../images/train/foo.png

    to:

        .../labels/train/foo.txt
    """
    parts = list(
        image_path.parts
    )

    indices = [
        i
        for i, part in enumerate(parts)
        if part == "images"
    ]

    if not indices:
        raise ValueError(
            "Cannot infer label path because "
            "the image path does not contain "
            "an 'images' directory:\n"
            f"{image_path}"
        )

    index = indices[-1]

    label_parts = (
        parts[:index]
        + ["labels"]
        + parts[index + 1 :]
    )

    return Path(
        *label_parts
    ).with_suffix(".txt")


def remap_label_file(
    source_label: Path,
    destination_label: Path,
    class_mapping: dict[int, int],
) -> None:
    """
    Copy an Ultralytics segmentation label while remapping
    class IDs.
    """
    if not source_label.exists():
        # A missing label is a valid background-only
        # image in Ultralytics datasets.
        destination_label.write_text(
            "",
            encoding="utf-8",
        )
        return

    output_lines: list[str] = []

    for line_number, line in enumerate(
        source_label.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        fields = line.split()

        # class + at least three x/y pairs
        if len(fields) < 7:
            raise ValueError(
                f"Invalid segmentation row in "
                f"{source_label}:{line_number}:\n"
                f"{line}"
            )

        old_class_id = int(
            fields[0]
        )

        if old_class_id not in (
            class_mapping
        ):
            raise ValueError(
                f"Unknown class ID "
                f"{old_class_id} in "
                f"{source_label}"
            )

        new_class_id = (
            class_mapping[
                old_class_id
            ]
        )

        output_lines.append(
            " ".join(
                [
                    str(new_class_id),
                    *fields[1:],
                ]
            )
        )

    destination_label.write_text(
        "\n".join(output_lines)
        + (
            "\n"
            if output_lines
            else ""
        ),
        encoding="utf-8",
    )


def merge_yolo_seg_datasets(
    yaml_files: Iterable[Path],
    output_root: Path,
) -> None:
    """
    Merge two or more Ultralytics segmentation datasets.

    Classes are matched by class NAME, not by their original
    numerical IDs.

    Example
    -------

    Dataset A:

        0: rods
        1: grids

    Dataset B:

        0: grids
        1: screws

    Merged:

        0: rods
        1: grids
        2: screws

    Dataset B's label files are rewritten accordingly.
    """
    yaml_files = [
        Path(path).resolve()
        for path in yaml_files
    ]

    if len(yaml_files) < 2:
        raise ValueError(
            "At least two datasets are "
            "required for merging."
        )

    output_root = (
        Path(output_root).resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = []

    global_names: list[str] = []

    # ------------------------------------------------------------------
    # Build global class mapping.
    # ------------------------------------------------------------------

    for yaml_path in yaml_files:
        with yaml_path.open(
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f)

        names = load_names(data)

        for name in names.values():
            if name not in global_names:
                global_names.append(name)

        datasets.append(
            (
                yaml_path,
                data,
                names,
            )
        )

    global_name_to_id = {
        name: i
        for i, name in enumerate(
            global_names
        )
    }

    found_splits: set[str] = set()

    # ------------------------------------------------------------------
    # Merge train / val / test.
    # ------------------------------------------------------------------

    for (
        dataset_index,
        (
            yaml_path,
            data,
            names,
        ),
    ) in enumerate(datasets):
        class_mapping = {
            old_id: global_name_to_id[
                name
            ]
            for old_id, name
            in names.items()
        }

        for split in (
            "train",
            "val",
            "test",
        ):
            images = (
                resolve_split_images(
                    yaml_path,
                    data,
                    split,
                )
            )

            if not images:
                continue

            found_splits.add(split)

            destination_images = (
                output_root
                / "images"
                / split
            )

            destination_labels = (
                output_root
                / "labels"
                / split
            )

            destination_images.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination_labels.mkdir(
                parents=True,
                exist_ok=True,
            )

            for (
                image_index,
                image_path,
            ) in enumerate(tqdm(images,desc=f'{yaml_path.parent.name} {split=}')):
                # Prefix names so files from separate
                # datasets can never overwrite each other.
                new_stem = (
                    f"d{dataset_index:03d}_"
                    f"{image_index:06d}_"
                    f"{image_path.stem}"
                )

                new_image_name = (
                    new_stem
                    + image_path.suffix.lower()
                )

                destination_image = (
                    destination_images
                    / new_image_name
                )

                destination_label = (
                    destination_labels
                    / f"{new_stem}.txt"
                )

                shutil.copy2(
                    image_path,
                    destination_image,
                )

                source_label = (
                    find_label_for_image(
                        image_path
                    )
                )

                remap_label_file(
                    source_label=source_label,
                    destination_label=destination_label,
                    class_mapping=class_mapping,
                )

    if not found_splits:
        raise ValueError(
            "No train/val/test images were "
            "found in the input datasets."
        )

    write_data_yaml(
        output_root=output_root,
        classes=global_names,
        splits=found_splits,
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert semantic segmentation datasets "
            "to Ultralytics instance segmentation format "
            "and merge Ultralytics segmentation datasets."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ----------------------------------------------------------------------
    # CamVid
    # ----------------------------------------------------------------------

    camvid_parser = (
        subparsers.add_parser(
            "camvid",
            help=(
                "Convert a CamVid-style "
                "semantic segmentation dataset."
            ),
        )
    )

    camvid_parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Root directory of the "
            "CamVid dataset."
        ),
    )

    camvid_parser.add_argument(
        "--split",
        action="append",
        type=parse_key_value,
        metavar="NAME=FILE",
        help=(
            "Optional explicit split manifest. "
            "Can be specified multiple times. "
            "Example: "
            "--split train=train.txt "
            "--split val=val.txt"
        ),
    )

    camvid_parser.add_argument(
        "--pairs",
        type=Path,
        help=(
            "Optional single unsplit CamVid "
            "image/mask manifest. "
            "It will become the train split."
        ),
    )

    camvid_parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        type=parse_key_value,
        required=True,
        metavar="NAME=MASK_VALUE",
        help=(
            "Class name and integer value in "
            "the semantic mask. "
            "Can be specified repeatedly. "
            "Example: --class rods=2"
        ),
    )

    camvid_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    camvid_parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help=(
            "Ignore connected components "
            "smaller than this number of "
            "pixels. Default: 1"
        ),
    )

    # ----------------------------------------------------------------------
    # Typed dataset
    # ----------------------------------------------------------------------

    typed_parser = (
        subparsers.add_parser(
            "typed",
            help=(
                "Convert sample directories "
                "containing one binary mask "
                "per class."
            ),
        )
    )

    typed_parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Root directory of the source "
            "dataset."
        ),
    )

    typed_parser.add_argument(
        "--split",
        action="append",
        type=parse_key_value,
        required=True,
        metavar="NAME=DIRECTORY",
        help=(
            "Directory containing sample "
            "folders. Example: "
            "--split train=train. "
            "Use --split train=. when sample "
            "folders are directly under --root."
        ),
    )

    typed_parser.add_argument(
        "--image-name",
        default="img.png",
        help=(
            "Image filename inside each "
            "sample directory. "
            "Default: img.png"
        ),
    )

    typed_parser.add_argument(
        "--class-mask",
        dest="class_masks",
        action="append",
        type=parse_key_value,
        required=True,
        metavar="NAME=FILENAME",
        help=(
            "Class name and corresponding "
            "binary mask filename. "
            "Can be specified repeatedly. "
            "Example: "
            "--class-mask rods=rods.png"
        ),
    )

    typed_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    typed_parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help=(
            "Ignore connected components "
            "smaller than this number of "
            "pixels. Default: 1"
        ),
    )

    # ----------------------------------------------------------------------
    # Merge
    # ----------------------------------------------------------------------

    merge_parser = (
        subparsers.add_parser(
            "merge",
            help=(
                "Merge two or more existing "
                "Ultralytics segmentation "
                "datasets."
            ),
        )
    )

    merge_parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Input data.yaml files."
        ),
    )

    merge_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    # ----------------------------------------------------------------------
    # Execute
    # ----------------------------------------------------------------------

    args = parser.parse_args()

    if args.command == "camvid":
        root = args.root.resolve()

        split_files = discover_camvid_splits(
            source_root=root,
            explicit_splits=args.split,
            pairs_file=args.pairs,
        )

        classes = {
            name: int(value)
            for name, value
            in args.classes
        }

        convert_camvid_dataset(
            source_root=root,
            split_files=split_files,
            output_root=args.output,
            classes=classes,
            min_area=args.min_area,
        )

    elif args.command == "typed":
        root = args.root.resolve()

        split_dirs: dict[str, Path] = {}

        for name, dirname in args.split:
            path = Path(dirname)

            if not path.is_absolute():
                path = root / path

            split_dirs[name] = (
                path.resolve()
            )

        class_masks = dict(
            args.class_masks
        )

        convert_typed_dataset(
            split_dirs=split_dirs,
            output_root=args.output,
            image_name=args.image_name,
            class_masks=class_masks,
            min_area=args.min_area,
        )

    elif args.command == "merge":
        merge_yolo_seg_datasets(
            yaml_files=args.data,
            output_root=args.output,
        )


if __name__ == "__main__":
    main()
