import seglight.image_utils as iu
from seglight.domain import Image


class ImgTilingResult:
    def __init__(self,tiles,xy,img_shape):
        self.tiles = tiles
        self.xy = xy
        self.img_shape = img_shape

def tile_image(img:Image, tile_size:int, overlap:int) -> ImgTilingResult:
    tiles,xy,overlap,img_shape = iu.tile_image_with_overlap(img,tile_size,overlap)

    return ImgTilingResult(
        tiles,
        xy,
        img_shape
    )

def blend_tiles(tile_result: ImgTilingResult)->Image:
    return iu.blend_tiles(
        tile_result.tiles,
        tile_result.xy,
        tile_result.img_shape
    )
