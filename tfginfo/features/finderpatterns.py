from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np
from skimage import color, filters

from tfginfo.utils import Image
from .ratios import find_general
from .models import FinderPattern


def find_finder_patterns_ratios(image: Image, block_size=151, **kwargs) -> List[FinderPattern]:
    gray_image: np.ndarray = color.rgb2gray(image)
    threshold: np.ndarray = filters.threshold_sauvola(gray_image, block_size)
    bw_image: np.ndarray = gray_image > threshold

    return list(map(
        lambda t: FinderPattern.from_center_and_ratios(
                image=bw_image,
                center=t[0],
                hratios=t[1],
                vratios=t[2]
        ),
        find_general(
            bw_image=bw_image,
            iratios=[1, 1, 1, 3, 1, 1, 1],
            center_color=False,
            strict_border=False
        )
    ))


def find_finder_patterns_areas(image: Image, **kwargs) -> List[FinderPattern]:
    pass


@unique
class FinderPatternMethods(Enum):
    CLASSIC = auto()
    AREAS = auto()
    # ZBAR = auto()
    # OPENCV = auto()


_FinderPatternFinder = Callable[..., List[FinderPattern]]
_FINDER_PATTERN_METHODS_FUNCS: Dict[FinderPatternMethods, _FinderPatternFinder] = {
    FinderPatternMethods.CLASSIC: find_finder_patterns_ratios,
    FinderPatternMethods.AREAS: find_finder_patterns_areas
}


def find_finder_patterns(image: Image,
                         method: Optional[FinderPatternMethods] = None,
                         **kwargs) -> List[FinderPattern]:
    if method is None:
        method = FinderPatternMethods.CLASSIC

    func = _FINDER_PATTERN_METHODS_FUNCS[method]
    return func(image, **kwargs)
