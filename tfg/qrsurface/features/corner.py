from typing import Optional

import numpy as np

from .utils import create_line_iterator


def rot90_point(point, k: int, shape) -> np.ndarray:
    """
    Rotate 90 degree's a point inside of a image.

    :param point: 2D point to rotate.
    :param k: Number of 90 degree's rotations to apply.
    :param shape: Shape of the image.

    :return: The point resulting from the rotation.

    """
    image_h, image_w = shape
    x, y = point
    for ii in range(k):
        x, y = -y, x

    return np.array((x, y), dtype=int) + np.array((image_w - 1 if k == 1 or k == 2 else 0,
                                                   image_h - 1 if k == 2 or k == 3 else 0), dtype=int)


def corner_filter(line: np.ndarray, center: int, corner_radius: int, fuzzy_radius: int, blank_radius: int) -> bool:
    """
    Given a line of pixels, the hipotetic center of a corner, and some parameters that configure the check, returns
    if there is a corner at that position or not.

    X := The central pixel in a iteration, that will be extracted as the corner. Will be checked as a activated pixel
    O := The pixels that will be check as activated
    _ := The pixels that will be check as not activated
    / := The pixels that won't be check

                     ///_____////OOOXOOO////_____/////
                        <─┬─><┬─><┬>
                          │   │   │
    blank_radius ─────────┘   │   │
    fuzzy_radius ─────────────┘   │
    corner_radius ────────────────┘

    :param line: An array of pixels that will be check for corners.
    :param center: The index of the pixel that will be check as the center of the corner.
    :param corner_radius: The radius of pixels that will be check as activated.
    :param fuzzy_radius: The radius of pixels that won't be check.
    :param blank_radius: The radius of pixels that will be check as not activated.

    :return: If the given line seems to have a corner or not.

    """
    inner_radius = corner_radius + fuzzy_radius + 1
    return (all(line[center + j] == 0 for j in range(-corner_radius, corner_radius + 1))
            and all(line[center - j] != 0 for j in range(inner_radius, inner_radius + blank_radius + 1))
            and all(line[center + j] != 0 for j in range(inner_radius, inner_radius + blank_radius + 1))
            and line[center] == 0)


def corner_scan(image_threshold: np.ndarray, corner: int = 0, corner_radius: int = 1,
                fuzzy_radius: int = 0, blank_radius: int = 0) -> Optional[np.ndarray]:
    """
    Method that given a image search for corners beginning from one of the four corners of the
    image.

    :param image_threshold: Image thresholded.
    :param corner: From which of the four corners of the image begin to search.
    :param corner_radius: The radius of pixels around a hipotetic corner in a diagonal line that will be check
        as activated.
    :param fuzzy_radius: The radius of pixels around a hipotetic corner in a diagonal line that won't be check.
    :param blank_radius: The radius of pixels around a hipotetic corner in a diagonal line that will be check
        as not activated.

    :return: The point found or None if not found.

    """
    assert isinstance(image_threshold, np.ndarray)
    assert isinstance(corner, int)
    assert isinstance(corner_radius, int)
    assert isinstance(fuzzy_radius, int)
    assert isinstance(blank_radius, int)
    assert len(image_threshold.shape) == 2
    assert 0 <= corner < 4

    total_radius = corner_radius + fuzzy_radius + blank_radius + 1
    image_threshold = np.rot90(image_threshold, k=corner)
    image_h, image_w = image_threshold.shape
    for k in range(2, min(image_h, image_w)):
        p1 = np.array([k, 0], dtype=int)
        p2 = np.array([0, k], dtype=int)
        points, line = create_line_iterator(p1, p2, image_threshold)

        for i in range(total_radius, len(line) - total_radius):
            if corner_filter(line, i, corner_radius, fuzzy_radius, blank_radius):
                point = np.array(points[i], dtype=np.int64)
                rotated_point = rot90_point(point, k=corner, shape=image_threshold.shape)
                return rotated_point  # we append the middle point to the list of found points
