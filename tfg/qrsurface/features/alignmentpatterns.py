from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np

from .ratios import find_general
from .models import AlignmentPattern


def find_alignment_patterns_ratios(bw_image: np.ndarray, **kwargs) -> List[AlignmentPattern]:
    """
    Find all the alignment patterns in a image given

    :param bw_image: The binarized image
    :param kwargs: Keyword arguments for future extensions

    :return: The list of alignment patterns found
    """
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


@unique
class AlignmentPatternMethods(Enum):
    """
    All the possible method for finding alignment patterns
    """
    CLASSIC = auto()


# Mapper from identifier of localization method to the callable
_AlignmentPatternFinder = Callable[..., List[AlignmentPattern]]
_FINDER_PATTERN_METHODS_FUNCS: Dict[AlignmentPatternMethods, _AlignmentPatternFinder] = {
    AlignmentPatternMethods.CLASSIC: find_alignment_patterns_ratios
}


def find_alignment_patterns(bw_image: np.ndarray,
                            method: Optional[AlignmentPatternMethods] = None,
                            **kwargs) -> List[AlignmentPattern]:
    """
    Find all the alignment patterns in a image given

    :param bw_image: The binarized image
    :param method: The identifier of the method used
    :param kwargs: Keyword arguments for future extensions

    :return: The list of alignment patterns found
    """
    if method is None:
        method = AlignmentPatternMethods.CLASSIC

    func = _FINDER_PATTERN_METHODS_FUNCS[method]
    return func(bw_image, **kwargs)
