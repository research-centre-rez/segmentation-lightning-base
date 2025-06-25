import warnings

from torch.functional import F

from seglight.domain import Image


def normalize_image(image: Image) -> Image:
    max_value = image.max()
    min_value = image.min()
    if max_value == min_value:
        warnings.warn("Image min == Image max. Doing nothing.")
        return image
    return (image - min_value) / (max_value - min_value)


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0: # noqa SIM108
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0: # noqa SIM108
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x
