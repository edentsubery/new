import numpy as np
from PIL import Image

from functools import partial


norm2 = partial(np.linalg.norm, ord=2)


def write_img(img, img_path):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(img_path)

