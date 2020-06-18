from dataclasses import dataclass
from enum import auto, Enum
from typing import List


class MatchingFeatures(Enum):
    """
    An enumeration of the possible types of features that we can use as references
    """
    FINDER_CENTERS = auto()
    FINDER_CORNERS = auto()
    ALIGNMENTS_CENTERS = auto()
    FOURTH_CORNER = auto()


@dataclass
class References:
    """
    A dataclass representing a set of reference points, with two helper flags of metadata
    """
    features: List[MatchingFeatures]
    alignments_found: List[bool]
    fourth_corner_found: bool
