import os
from pathlib import Path

import imageio
import numpy as np


def read_camvid_pairs(path: os.PathLike) -> list[tuple[os.PathLike, os.PathLike]]:
    """
    Read CamVid file pairs from a text file.

    Each line in the file is expected to contain two paths separated by ' ',
    referring to an image and its corresponding label file. This format is used
    in datasets like CamVid.

    Example line in the file:
        default/0001TP_006690.png /defaultannot/0001TP_006690_L.png

    Parameters
    ----------
    path : os.PathLike
        Path to the `.txt` file containing the dataset file pairs.

    Returns
    -------
    list of tuple of os.PathLike
        A list of (image_path, label_path) pairs with leading slashes removed.
    """
    with open(path) as f:
        sep = " "  # additinal stripping of '/' to make paths relative
        parts = (line.strip().split(sep) for line in f)
        return [[Path(pp.lstrip("/")) for pp in p] for p in parts if len(p) == 2]


def create_pairs_paths(
    sample_names,
    image_destination="default",
    annotation_destination="defaultannot",
    file_ext=".png",
):
    dataset_structure = {}
    for sample_name in sample_names:
        dataset_structure[sample_name] = {
            "image_path": f"/{image_destination}/{sample_name}{file_ext}",
            "annotation_path": f"{annotation_destination}/{sample_name}{file_ext}",
        }
    return dataset_structure


def structure_dataset_text(dataset_structure):
    lines = [
        f"{v['image_path']} {v['annotation_path']}\n"
        for v in dataset_structure.values()
    ]
    return "".join(lines)


def _merge_rgb_masks(rgb_masks):
    base = np.zeros_like(rgb_masks[0], dtype=int)

    for rgb_mask in rgb_masks:
        bin_mask = np.sum(rgb_mask, axis=2) > 0
        base[bin_mask] = rgb_mask[bin_mask]
    return base


def color_masks(label_color_mapping, img_dict):
    masks = []
    for label_name, rgb_color in label_color_mapping.items():
        binary_mask = img_dict[label_name]
        rgb = np.expand_dims(binary_mask, axis=2) * np.array(rgb_color)
        masks.append(rgb)
    return masks


def collect_camvid_pairs(data_dict, label_color_mapping, img_key):
    rgb_masks = {}
    imgs = {}
    for k, img_dict in data_dict.items():
        masks = color_masks(label_color_mapping, img_dict)
        rgb_masks[k] = _merge_rgb_masks(masks)
        imgs[k] = np.uint8(np.dstack([img_dict[img_key] * 255] * 3))
    return imgs, rgb_masks


class CamVidDSFormat:
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
            img = imageio.imread(dr / str(img_p))
            lbl_ch = imageio.imread(dr / str(lbl_p))

            lbls = _colors_to_masks(lbl_ch, label_dict)

            data[name] = {"img": img} | lbls
        return data


def _read_label_map(label_path):
    with open(label_path) as f:
        l_split = (line.split() for line in f)
        return {lbl_id: [int(r), int(g), int(b)] for r, g, b, lbl_id in l_split}


def _get_color_as_bin(lbl, color):
    bool_mask = lbl == np.array(color)[None, None]
    color_match = np.all(bool_mask, axis=2)
    return np.float32(color_match)


def _colors_to_masks(lbl, labels_dict):
    return {
        label_id: _get_color_as_bin(lbl, col) for label_id, col in labels_dict.items()
    }


def _colorize_gs_image(image, color_rgb: list[int]):
    color = np.array(color_rgb)
    img_rgb = np.dstack([image] * 3) * color
    return np.uint8(img_rgb)
