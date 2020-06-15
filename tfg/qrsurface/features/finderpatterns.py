from enum import auto, Enum, unique
from typing import Callable, Dict, List, Optional

import numpy as np

from .ratios import find_general
from .models import FinderPattern


def find_finder_patterns_ratios(bw_image: np.ndarray, **kwargs) -> List[FinderPattern]:
    return list(map(
        lambda t: FinderPattern.from_center_and_ratios(
            image=bw_image,
            center=t[0],
            hratios=t[1],
            vratios=t[2]
        ),
        find_general(
            bw_image=bw_image,
            iratios=[1, 1, 3, 1, 1],#[1, 1, 1, 3, 1, 1, 1],
            center_color=False,
            strict_border=True,#False
            diagonals=True,
            countours=False
        )
    ))


def find_finder_patterns_areas(bw_image: np.ndarray, **kwargs) -> List[FinderPattern]:
    pass


@unique
class FinderPatternMethods(Enum):
    CLASSIC = auto()
    AREAS = auto()


_FinderPatternFinder = Callable[..., List[FinderPattern]]
_FINDER_PATTERN_METHODS_FUNCS: Dict[FinderPatternMethods, _FinderPatternFinder] = {
    FinderPatternMethods.CLASSIC: find_finder_patterns_ratios,
    FinderPatternMethods.AREAS: find_finder_patterns_areas
}


def find_finder_patterns(bw_image: np.ndarray,
                         method: Optional[FinderPatternMethods] = None,
                         **kwargs) -> List[FinderPattern]:
    if method is None:
        method = FinderPatternMethods.CLASSIC

    func = _FINDER_PATTERN_METHODS_FUNCS[method]
    return func(bw_image, **kwargs)
