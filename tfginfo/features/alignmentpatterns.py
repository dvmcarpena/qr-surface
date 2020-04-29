from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np
from skimage import color, filters

from tfginfo.utils import Image
from .ratios import find_general
from .models import AlignmentPattern


def find_alignment_patterns_ratios(image: Image, block_size=151, **kwargs) -> List[AlignmentPattern]:
    gray_image: np.ndarray = color.rgb2gray(image)
    threshold: np.ndarray = filters.threshold_sauvola(gray_image, block_size)
    bw_image: np.ndarray = gray_image > threshold

    return list(map(
        lambda t: AlignmentPattern.from_center_and_ratios(
                image=bw_image,
                center=t[0],
                hratios=t[1],
                vratios=t[2]
        ),
        find_general(
            bw_image=bw_image,
            iratios=[1, 1, 1, 1, 1],
            center_color=False,
            strict_border=False,
            diagonals=True,
            countours=False
        )
    ))


def find_alignment_patterns_areas(image: Image, **kwargs) -> List[AlignmentPattern]:
    pass


@unique
class AlignmentPatternMethods(Enum):
    CLASSIC = auto()
    AREAS = auto()


_AlignmentPatternFinder = Callable[..., List[AlignmentPattern]]
_FINDER_PATTERN_METHODS_FUNCS: Dict[AlignmentPatternMethods, _AlignmentPatternFinder] = {
    AlignmentPatternMethods.CLASSIC: find_alignment_patterns_ratios,
    AlignmentPatternMethods.AREAS: find_alignment_patterns_areas
}


def find_alignment_patterns(image: Image,
                            method: Optional[AlignmentPatternMethods] = None,
                            **kwargs) -> List[AlignmentPattern]:
    if method is None:
        method = AlignmentPatternMethods.CLASSIC

    func = _FINDER_PATTERN_METHODS_FUNCS[method]
    return func(image, **kwargs)
