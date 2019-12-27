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


def get_num_aligns_from_version(version: int) -> int:
    return _NUM_ALIGNMENT_PATTERN_BY_VERSION[version - 1]
