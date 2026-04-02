import numpy as np
from dataclasses import dataclass

@dataclass
class BoundingBox:
    top:int
    bottom:int
    left:int
    right:int

    @staticmethod
    def from_mask(mask):
        t,b,l,r = bbox(mask)
        return BoundingBox(t,b,l,r)
    
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (ymin, ymax + 1, xmin, xmax + 1)
    
