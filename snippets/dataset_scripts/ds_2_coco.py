#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


def load_instance_mask(mask_path: Path) -> np.ndarray:
    """
    Load instance mask.

    Supports:
    - 2D grayscale masks
    - RGB/RGBA masks where R == G == B
    """
    with Image.open(mask_path) as im:
        arr = np.array(im)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        rgb = arr[..., :3]

        if not (
            np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 0], rgb[..., 2])
        ):
            raise ValueError(
                f"{mask_path} has multiple channels, but they are not identical grayscale values."
            )

        return rgb[..., 0]

    raise ValueError(f"{mask_path} has unsupported mask shape {arr.shape}")


def binary_mask_to_polygons(mask: np.ndarray):
    """
    Convert a binary mask to polygons.

    Returns polygons in pixel coordinates:
    [
        [x1, y1, x2, y2, ...],
        ...
    ]
    """
    mask = mask.astype(np.uint8)

    # Padding prevents border-touching objects from losing contours.
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded, level=0.5)

    polygons = []

    for contour in contours:
        # Remove padding offset.
        contour -= 1

        polygon = []

        # skimage gives (row, col) == (y, x)
        for y, x in contour:
            polygon.extend([float(x), float(y)])

        # At least 3 points.
        if len(polygon) >= 6:
            polygons.append(polygon)

    return polygons


def clip_polygon_to_image(polygon, width: int, height: int):
    clipped = []

    for i in range(0, len(polygon), 2):
        x = float(np.clip(polygon[i], 0, width - 1))
        y = float(np.clip(polygon[i + 1], 0, height - 1))
        clipped.extend([x, y])

    return clipped


def polygon_to_yolo_line(polygon, width: int, height: int, class_id: int):
    """
    Convert pixel-coordinate polygon to one YOLO segmentation line.

    YOLO segmentation format:
    class_id x1 y1 x2 y2 ...
    with coordinates normalized to [0, 1].
    """
    values = [str(class_id)]

    for i in range(0, len(polygon), 2):
        x = polygon[i] / width
        y = polygon[i + 1] / height

        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))

        values.append(f"{x:.6f}")
        values.append(f"{y:.6f}")

    return " ".join(values)


def binary_mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def binary_mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return [
        x_min,
        y_min,
        x_max - x_min + 1,
        y_max - y_min + 1,
    ]


def collect_samples(
    input_dir: Path,
    image_filename: str,
    mask_filename: str,
):
    samples = []

    for sample_dir in sorted(input_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        image_path = sample_dir / image_filename
        mask_path = sample_dir / mask_filename

        if not image_path.exists():
            print(f"Skipping {sample_dir}: missing image {image_filename}")
            continue

        if not mask_path.exists():
            print(f"Skipping {sample_dir}: missing mask {mask_filename}")
            continue

        samples.append(
            {
                "sample_name": sample_dir.name,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )

    return samples


def split_samples(samples, val_fraction: float, seed: int):
    samples = list(samples)

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_val = int(round(len(samples) * val_fraction))

    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    return train_samples, val_samples


def prepare_image(
    source_image_path: Path,
    target_image_path: Path,
    copy_images: bool,
):
    target_image_path.parent.mkdir(parents=True, exist_ok=True)

    if copy_images:
        shutil.copy2(source_image_path, target_image_path)
    else:
        if target_image_path.exists() or target_image_path.is_symlink():
            target_image_path.unlink()

        target_image_path.symlink_to(source_image_path.resolve())


def convert_split(
    samples,
    split_name: str,
    output_dir: Path,
    image_filename: str,
    category_name: str,
    copy_images: bool,
):
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    image_id = 1

    for sample in samples:
        sample_name = sample["sample_name"]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        with Image.open(image_path) as img:
            width, height = img.size

        instance_mask = load_instance_mask(mask_path)

        if instance_mask.shape[:2] != (height, width):
            raise ValueError(
                f"Image/mask size mismatch in {sample_name}: "
                f"image has {(height, width)}, mask has {instance_mask.shape[:2]}"
            )

        target_image_name = f"{sample_name}{image_path.suffix}"
        target_image_path = images_dir / target_image_name
        target_label_path = labels_dir / f"{sample_name}.txt"

        prepare_image(
            source_image_path=image_path,
            target_image_path=target_image_path,
            copy_images=copy_images,
        )

        instance_ids = [int(v) for v in np.unique(instance_mask) if int(v) != 0]

        yolo_lines = []

        coco_images.append(
            {
                "id": image_id,
                "file_name": f"images/{split_name}/{target_image_name}",
                "width": width,
                "height": height,
            }
        )

        for instance_id in instance_ids:
            binary = instance_mask == instance_id

            area = binary_mask_area(binary)
            if area == 0:
                continue

            bbox = binary_mask_bbox(binary)
            if bbox is None:
                continue

            polygons = binary_mask_to_polygons(binary)

            valid_polygons = []

            for polygon in polygons:
                polygon = clip_polygon_to_image(polygon, width=width, height=height)

                if len(polygon) < 6:
                    continue

                valid_polygons.append(polygon)

                yolo_lines.append(
                    polygon_to_yolo_line(
                        polygon=polygon,
                        width=width,
                        height=height,
                        class_id=0,
                    )
                )

            if not valid_polygons:
                print(
                    f"Warning: no valid polygon for instance {instance_id} "
                    f"in {mask_path}"
                )
                continue

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": valid_polygons,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        target_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        image_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "object",
            }
        ],
    }

    return coco


def write_yaml(output_dir: Path, category_name: str):
    yaml_path = output_dir / "data.yaml"

    yaml_text = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: {category_name}
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    image_filename: str,
    mask_filename: str,
    category_name: str,
    val_fraction: float,
    seed: int,
    copy_images: bool,
):
    samples = collect_samples(
        input_dir=input_dir,
        image_filename=image_filename,
        mask_filename=mask_filename,
    )

    if not samples:
        raise RuntimeError(f"No valid samples found in {input_dir}")

    train_samples, val_samples = split_samples(
        samples=samples,
        val_fraction=val_fraction,
        seed=seed,
    )

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    coco_train = convert_split(
        samples=train_samples,
        split_name="train",
        output_dir=output_dir,
        image_filename=image_filename,
        category_name=category_name,
        copy_images=copy_images,
    )

    coco_val = convert_split(
        samples=val_samples,
        split_name="val",
        output_dir=output_dir,
        image_filename=image_filename,
        category_name=category_name,
        copy_images=copy_images,
    )

    train_json = annotations_dir / "instances_train.json"
    val_json = annotations_dir / "instances_val.json"

    train_json.write_text(json.dumps(coco_train, indent=2), encoding="utf-8")
    val_json.write_text(json.dumps(coco_val, indent=2), encoding="utf-8")

    yaml_path = write_yaml(
        output_dir=output_dir,
        category_name=category_name,
    )

    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Ultralytics YAML: {yaml_path}")
    print(f"COCO train JSON: {train_json}")
    print(f"COCO val JSON: {val_json}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert foldered image + instance mask dataset into "
            "Ultralytics YOLO segmentation format and COCO JSON annotations."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input dataset root containing sample folders.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output YOLO/COCO dataset directory.",
    )

    parser.add_argument(
        "--image-filename",
        default="img.png",
        help="Image filename inside each sample folder.",
    )

    parser.add_argument(
        "--mask-filename",
        default="instance_mask.png",
        help="Instance mask filename inside each sample folder.",
    )

    parser.add_argument(
        "--category-name",
        default="object",
        help="Single class name.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help=(
            "Copy images instead of symlinking them. "
            "Use this if your training environment does not like symlinks."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_filename=args.image_filename,
        mask_filename=args.mask_filename,
        category_name=args.category_name,
        val_fraction=args.val_fraction,
        seed=args.seed,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
