from dataclasses import dataclass
from enum import auto, Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import imageio
import numpy as np
from skimage import img_as_ubyte, color, filters

from tfginfo.error import QRErrorId


class Deformation(Enum):
    PROJECTIVE = auto()
    CYLINDRIC = auto()
    SURFACE = auto()


@dataclass
class LabeledImage:
    path: Path
    original_id: id
    version: int
    has_data: bool
    method: str
    num_qrs: int
    error_id: Optional[QRErrorId]

    @classmethod
    def from_path(cls, path: Path, root: Path) -> 'LabeledImage':
        version, method, num_qrs, status, *rest = path.relative_to(root).parts
        return cls(
            path=path,
            version=int(version.split("_")[0].replace("v", "")),
            original_id=version.split("_")[1],
            has_data=(root / (version.split("_")[1] + ".png")).exists(),
            method=Deformation.__members__[method.upper()],
            num_qrs=int(num_qrs),
            error_id=QRErrorId.__members__[rest[0]] if status != "good" else None
        )

    def has_error(self) -> bool:
        return self.error_id is not None


def read_original_matrix(original_path: Path) -> np.ndarray:
    original_path = imageio.imread(original_path)
    gray_image = color.rgb2gray(original_path)
    threshold = filters.threshold_otsu(gray_image)
    return img_as_ubyte(color.gray2rgb(gray_image > threshold))


def parse_original_qrs(labeled_images_dir: Path) -> Dict[str, np.ndarray]:
    return {
        p.stem: read_original_matrix(p)
        for p in labeled_images_dir.iterdir()
        if p.is_file()
    }


def parse_labeled_images(labeled_images_dir: Path,
                         filter_func: Optional[Callable[[LabeledImage], bool]] = None) -> List[LabeledImage]:
    return list(filter(
        filter_func if filter_func is not None else lambda _: True,
        map(
            lambda p: LabeledImage.from_path(p, labeled_images_dir),
            filter(
                lambda p: p.is_file() and p.parent != labeled_images_dir,
                labeled_images_dir.glob("**/*")
            )
        )
    ))
