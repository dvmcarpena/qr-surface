import itertools
from typing import List, Tuple

import numpy as np

Array = np.ndarray
Image = np.ndarray

_NUM_ALIGNMENT_PATTERN_BY_VERSION = [0] + [
    1
    for _ in range(2, 7)
] + [
    6
    for _ in range(7, 14)
] + [
    13
    for _ in range(14, 21)
] + [
    22
    for _ in range(21, 28)
] + [
    33
    for _ in range(28, 35)
] + [
    46
    for _ in range(35, 41)
]

_ALIGNMENT_PATTERN_POSITIONS_TABLE = [
    [],
    [6, 18],
    [6, 22],
    [6, 26],
    [6, 30],
    [6, 34],
    [6, 22, 38],
    [6, 24, 42],
    [6, 26, 46],
    [6, 28, 50],
    [6, 30, 54],
    [6, 32, 58],
    [6, 34, 62],
    [6, 26, 46, 66],
    [6, 26, 48, 70],
    [6, 26, 50, 74],
    [6, 30, 54, 78],
    [6, 30, 56, 82],
    [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94],
    [6, 26, 50, 74, 98],
    [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106],
    [6, 32, 58, 84, 110],
    [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122],
    [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130],
    [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138],
    [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150],
    [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158],
    [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166],
    [6, 30, 58, 86, 114, 142, 170]
]


def get_num_aligns_from_version(version: int) -> int:
    return _NUM_ALIGNMENT_PATTERN_BY_VERSION[version - 1]


def get_alignment_pattern_positions(version: int) -> List[int]:
    return _ALIGNMENT_PATTERN_POSITIONS_TABLE[version - 1]


def get_size_from_version(version: int) -> int:
    return version * 4 + 17


def get_alignments_centers(version: int) -> List[Tuple[int, int]]:
    """
    Gets the centers of the alignment table.

    :param version: The version.

    :return: The centers of the alignment table.
    """
    if version == 1:
        return []

    positions = get_alignment_pattern_positions(version)
    all_coords = list(itertools.product(positions, positions))

    all_coords.remove((positions[0], positions[0]))
    all_coords.remove((positions[-1], positions[0]))
    all_coords.remove((positions[0], positions[-1]))

    return all_coords


def create_bounding_box(points: Array) -> Array:
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    return np.array([(min_x, min_y), (max_x, max_y)])
