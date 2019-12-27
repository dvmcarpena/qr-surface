from typing import List, Optional, Tuple

import numpy as np
from skimage import draw, feature, measure

from tfginfo.utils import Array, Image


def get_center_from_contour(contour: Array, shape: Tuple[int, int]) -> Array:
    # TODO optimize to bbox only
    cimg = np.zeros(shape)
    cimg[contour.astype(np.uint32)] = 1
    m = measure.moments(cimg)
    return np.array([m[1, 0] / m[0, 0], m[0, 1] / m[0, 0]])


def get_corners_from_contour(contour: Array,
                             shape: Tuple[int, int],
                             num_corners: int,
                             min_distance: Optional[int] = None) -> Array:
    # TODO optimize to bbox only
    rimg = np.zeros(shape)
    rr, cc = draw.polygon(contour.T[0], contour.T[1], rimg.shape)
    rimg[rr, cc] = 1

    corners_measure = feature.corner_harris(rimg, sigma=3)
    kwargs = dict(num_peaks=num_corners)
    if min_distance is not None:
        kwargs.update(min_distance=min_distance)
    return feature.corner_peaks(corners_measure, **kwargs)[:, ::-1]


def get_ratios_from_center(image: Image, center: Array) -> Tuple[Array, Array]:
    pass
    # TODO


def get_contour_from_center_and_ratios(image: Image, center: Array, hratios: Array,
                                       vratios: Array, test_point_xoffset: int) -> Array:
    hsize, vsize = sum(hratios), sum(vratios)
    radius = 0.75
    hradius, vradius = int(hsize * radius), int(vsize * radius)
    heps, veps = sum(hratios[1:4]) // 2, sum(vratios[1:4]) // 2

    fimg_rect = (
        (center[0] - hradius, center[1] - vradius),
        (center[0] + hradius, center[1] + vradius)
    )
    fimg = image[fimg_rect[0][0]:fimg_rect[1][0], fimg_rect[0][1]:fimg_rect[1][1]]
    fimg_center = np.array([hradius, vradius])
    fimg_to_image = np.array([
        center[0] - fimg_center[0],
        center[1] - fimg_center[1]
    ])

    contours: List[np.ndarray] = measure.find_contours(fimg, 0.8)
    candidates = []
    for contour in contours:
        centroid = get_center_from_contour(contour, fimg.shape)
        is_finder_contour = (
                abs(centroid[0] - fimg_center[0]) < heps
                and abs(centroid[1] - fimg_center[1]) < veps
        )

        test_point = [[fimg_center[0],
                       fimg_center[1] - test_point_xoffset]]
        is_exterior_contour = measure.points_in_poly(test_point, contour)[0]

        if is_finder_contour and is_exterior_contour:
            candidates.append(contour)

    if len(candidates) > 1:
        raise ValueError("More than one contour found")
    if len(candidates) == 0:
        raise ValueError("No contour found")
    contour = candidates[0]
    contour += fimg_to_image
    # TODO maybe centroid is better center?
    return contour
