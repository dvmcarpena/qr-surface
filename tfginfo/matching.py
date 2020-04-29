from dataclasses import dataclass
from enum import auto, Enum
from typing import List


class MatchingFeatures(Enum):
    FINDER_CENTERS = auto()
    FINDER_CORNERS = auto()
    ALIGNMENTS_CENTERS = auto()
    FOURTH_CORNER = auto()


@dataclass
class References:
    features: List[MatchingFeatures]
    alignments_found: List[bool]
    fourth_corner_found: bool
