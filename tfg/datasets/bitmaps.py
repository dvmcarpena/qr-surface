from pathlib import Path
from typing import Dict

import imageio
import numpy as np
from skimage import img_as_ubyte, color, filters

from .images import LabeledImage

BitmapImage = np.ndarray


def read_bitmap(original_path: Path) -> BitmapImage:
    """
    Read the original bitmap image from the filesystem

    :param original_path: Path to the original bitmap

    :return: The bitmap original matrix
    """
    original_path = imageio.imread(original_path)
    gray_image = color.rgb2gray(original_path)
    threshold = filters.threshold_otsu(gray_image)
    return img_as_ubyte(color.gray2rgb(gray_image > threshold))


class BitmapCollection:
    """
    Collection of all the original bitmaps of the datasets
    """

    def __init__(self, labeled_images_dir: Path):
        self.bitmaps: Dict[str, BitmapImage] = {
            f"{p.stem}_{f.stem}": read_bitmap(f)
            for p in labeled_images_dir.iterdir()
            if p.is_dir()
            for f in (p / "bitmaps").iterdir()
            if f.is_file() and f.suffix == ".png"
        }

    def get_bitmap(self, labeled_image: LabeledImage) -> BitmapImage:
        """
        Given a labeled image, return its original bitmap matrix

        :param labeled_image: A labeled image

        :return: The original bitmap matrix
        """
        return self.bitmaps[f"{labeled_image.dataset}_{labeled_image.bitmap_id}"]
