from typing import List, Tuple, Union

import numpy as np
from skimage import draw, feature, measure

from ..utils import create_bounding_box


def get_center_from_contour(contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Find the center of a contour given

    :param contour: A array of 2D points that form a contour
    :param shape: The shape of the destiny image

    :return: Coordinates of the found center
    """
    cimg = np.zeros(shape)
    rr, cc = draw.polygon(contour.T[0], contour.T[1], cimg.shape)
    cimg[rr, cc] = 1
    m = measure.moments(cimg)

    if m[0, 0] < 1e-4:
        raise ValueError("The countour is too small")
    return np.array([m[1, 0] / m[0, 0], m[0, 1] / m[0, 0]])


def create_line_iterator(p1: np.ndarray, p2: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    :param p1: The first point
    :param p2: The second point
    :param img: The image being processed

    :return: Coordinates and intensities of each pixel
    """
    image_h = img.shape[0]
    image_w = img.shape[1]
    p1_x = p1[0]
    p1_y = p1[1]
    p2_x = p2[0]
    p2_y = p2[1]

    d_x = p2_x - p1_x
    d_y = p2_y - p1_y
    d_xa = np.abs(d_x).astype(np.int64)
    d_ya = np.abs(d_y).astype(np.int64)

    it_buffer = np.empty(shape=(np.maximum(d_ya, d_xa), 3), dtype=np.float32)
    it_buffer.fill(np.nan)

    neg_y = p1_y > p2_y
    neg_x = p1_x > p2_x
    if p1_x == p2_x:
        it_buffer[:, 0] = p1_x
        if neg_y:
            it_buffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
        else:
            it_buffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
    elif p1_y == p2_y:
        it_buffer[:, 1] = p1_y
        if neg_x:
            it_buffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
        else:
            it_buffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
    else:
        steep_slope = d_ya > d_xa
        if steep_slope:
            slope = d_x.astype(np.float32) / d_y.astype(np.float32)
            if neg_y:
                it_buffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
            else:
                it_buffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
            it_buffer[:, 0] = (slope * (it_buffer[:, 1] - p1_y)).astype(np.int) + p1_x
        else:
            slope = d_y.astype(np.float32) / d_x.astype(np.float32)
            if neg_x:
                it_buffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
            else:
                it_buffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
            it_buffer[:, 1] = (slope * (it_buffer[:, 0] - p1_x)).astype(np.int) + p1_y

    col_x = it_buffer[:, 0]
    col_y = it_buffer[:, 1]
    it_buffer = it_buffer[(col_x >= 0) & (col_y >= 0) & (col_x < image_w) & (col_y < image_h)]

    line_color = img[it_buffer[:, 1].astype(np.uint), it_buffer[:, 0].astype(np.uint)]

    return it_buffer[:, 0:2].astype(np.uint64), line_color.astype(np.uint64)


def get_corners_from_contour(contour: np.ndarray,
                             center: np.ndarray,
                             num_corners: int,
                             min_distance: int) -> np.ndarray:
    """
    Given a contour which we supose that has as many corners as given by the parameters num_corners, returns an array
    with the cordinates of that corners

    :param contour: A array of 2D points that form a contour
    :param center: The center of the contour
    :param num_corners: The number of corners to find
    :param min_distance: The minimum distance between corners

    :return: Array containing all the 2D corners
    """
    acontour = contour.astype(int)
    bbox = np.array([
        [acontour[:, 0].min(), acontour[:, 1].min()],
        [acontour[:, 0].max(), acontour[:, 1].max()]
    ])
    bbox_rad = max(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
    bbox[0, :] -= bbox_rad // 10
    bbox[1, :] += bbox_rad // 10
    nshape = (bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
    rimg = np.zeros(nshape)
    ncontour = contour - np.array(bbox[0])
    ncenter = center - np.array(bbox[0])
    rr, cc = draw.polygon(ncontour.T[0], ncontour.T[1], nshape)
    rimg[rr, cc] = 1

    corners_measure = feature.corner_harris(rimg, sigma=min_distance / 15)

    kwargs = dict(
        num_peaks=num_corners,
        min_distance=min_distance // 3,
        exclude_border=bbox_rad // (10 * 2)
    )
    corners = feature.peak_local_max(corners_measure, **kwargs)

    for i, c in enumerate(corners):
        vert = c[0] - ncenter[0] > 1
        if vert:
            m = (c[1] - ncenter[1]) / (c[0] - ncenter[0])
            n = c[1] - m * c[0]

            if c[0] < ncenter[0]:
                new_x = c[0] - min_distance
            else:
                new_x = c[0] + min_distance
            new_y = m * new_x + n
        else:
            m = (c[0] - ncenter[0]) / (c[1] - ncenter[1])
            n = c[0] - m * c[1]

            if c[1] < ncenter[1]:
                new_y = c[1] - min_distance
            else:
                new_y = c[1] + min_distance
            new_x = m * new_y + n

        coords, vals = create_line_iterator(c[::-1], np.array([new_x, new_y])[::-1], rimg)
        index = np.nonzero(vals == 0)[0][0]
        corners[i] = coords[index - 1][::-1]

    corners += np.array(bbox[0])

    return corners


def get_contours_with_center(image: np.ndarray, center: np.ndarray, hratios: Union[np.ndarray, List],
                             vratios: Union[np.ndarray, List]) -> List[List[np.ndarray]]:
    """
    Given a center of a finder or alignment pattern, finds the contours of that pattern

    :param image: Image where we want to find the contours
    :param center: The centers found of the patterns
    :param hratios: Horizontal ratios found
    :param vratios: Vertical ratios found

    :return: List of all the contours found, with the bounding box of the contour, center and contour
    """
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
    # fimg_to_image = np.array([
    #     center[0] - fimg_center[0],
    #     center[1] - fimg_center[1]
    # ])

    contours: List[np.ndarray] = measure.find_contours(fimg, 0.8)
    candidates = []
    for contour in contours:
        centroid = get_center_from_contour(contour, fimg.shape)
        is_finder_contour = (
                abs(centroid[0] - fimg_center[0]) < heps
                and abs(centroid[1] - fimg_center[1]) < veps
        )

        # test_point = [
        #     [fimg_center[0], fimg_center[1] - test_point_xoffset],
        #     [fimg_center[0], fimg_center[1] + test_point_xoffset],
        #     [fimg_center[0] - test_point_xoffset, fimg_center[1]],
        #     [fimg_center[0] + test_point_xoffset, fimg_center[1]]
        # ]
        # is_exterior_contour = all(measure.points_in_poly(test_point, contour))

        if is_finder_contour:  # and is_exterior_contour:
            bbox = create_bounding_box(contour)
            # contour += fimg_to_image
            candidates.append([bbox, contour, centroid])

    return candidates


def get_contour_from_center_and_ratios(image: np.ndarray, center: np.ndarray, hratios: np.ndarray,
                                       vratios: np.ndarray) -> np.ndarray:
    """
    Given a center of a finder or alignment pattern, finds the contour of that pattern

    :param image: Image where we want to find the contours
    :param center: The centers found of the patterns
    :param hratios: Horizontal ratios found
    :param vratios: Vertical ratios found

    :return: List of all the contours found, with the bounding box of the contour, center and contour
    """
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
        try:
            centroid = get_center_from_contour(contour, fimg.shape)
        except ValueError:
            continue
        is_finder_contour = (
                abs(centroid[0] - fimg_center[0]) < heps
                and abs(centroid[1] - fimg_center[1]) < veps
        )

        # test_point = [
        #     [fimg_center[0], fimg_center[1] - test_point_xoffset],
        #     [fimg_center[0], fimg_center[1] + test_point_xoffset],
        #     [fimg_center[0] - test_point_xoffset, fimg_center[1]],
        #     [fimg_center[0] + test_point_xoffset, fimg_center[1]]
        # ]
        # is_exterior_contour = all(measure.points_in_poly(test_point, contour))

        if is_finder_contour:  # and is_exterior_contour:
            bbox = create_bounding_box(contour)
            candidates.append([bbox, contour])

    m = None
    contour = None
    for bbox, c in candidates:
        area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
        if m is None:
            m = area
            contour = c
        elif area > m:
            m = area
            contour = c

    if contour is None:
        raise ValueError("No contour found")

    contour += fimg_to_image
    return contour
