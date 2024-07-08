import numpy as np
from PIL import Image
from skimage.draw import ellipse


def horizontal_mask(size: tuple[int, int], t0: float = 0.4, t1: float = 0.6,
                    w0: float = 0.1, w1: float = 0.1, invert: bool = False) -> Image:
    mask = np.empty((size[1], size[0]))
    mask.fill(255 if not invert else 0)
    start_indices = np.round(np.linspace(t0, t1, size[1]) * size[0]).astype(int)
    end_indices = np.round(np.linspace(t0 + w0, t1 + w1, size[1]) * size[0]).astype(int)
    for idx in range(len(start_indices)):
        mask[start_indices[idx]:end_indices[idx], idx] = 0 if not invert else 255
    return Image.fromarray(mask).convert('L')


def vertical_mask(size: tuple[int, int], t0: float = 0.4, t1: float = 0.6,
                  w0: float = 0.1, w1: float = 0.1, invert: bool = False) -> Image:
    mask = np.empty((size[1], size[0]))
    mask.fill(255 if not invert else 0)
    start_indices = np.round(np.linspace(t0, t1, size[0]) * size[1]).astype(int)
    end_indices = np.round(np.linspace(t0 + w0, t1 + w1, size[0]) * size[1]).astype(int)
    for idx in range(len(start_indices)):
        mask[idx, start_indices[idx]:end_indices[idx]] = 0 if not invert else 255
    return Image.fromarray(mask).convert('L')


def ellipse_mask(size: tuple[int, int], x: float, y: float, rx: float, ry: float, rotation: float = 0,
                 invert: bool = False) -> Image:
    mask = np.empty((size[1], size[0]))
    mask.fill(255 if not invert else 0)
    rr, cc = ellipse(int(y * size[1]), int(x * size[0]), int(ry * size[1]), int(rx * size[0]),
                     shape=mask.shape, rotation=np.deg2rad(rotation))
    mask[rr, cc] = 0 if not invert else 255
    return Image.fromarray(mask).convert('L')
