from tqdm.cli import tqdm
import pyvips
from pathlib import Path

src_dir = Path('<add_path>')
dst_dir = Path('<add_path>')
for p in tqdm(list(src_dir.rglob("*.png"))):
    img = pyvips.Image.new_from_file(str(p), access="sequential")  # sequential read is fine for conversion
    img = img.cast("float")
    # Normalize to 0–1 (for 8-bit PNG)
    img = img / 255.0
    # tile=True makes it tiled; tile_width/height choose tile size (256 or 512 are common)
    tiff_dir = dst_dir /p.parent.name
    tiff_dir.mkdir(exist_ok=True)
    img.tiffsave(
        str(tiff_dir /(p.stem + ".tif")),
        tile=True,
        tile_width=512,
        tile_height=512,
        compression="lzw",   # or "deflate"
        pyramid=False,       # set True if you want multires pyramids (bigger files, faster zooming/preview)
        bigtiff=True         # safe for large outputs
    )
