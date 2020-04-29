from typing import List, Optional, Tuple

import numpy as np
from skimage import draw, feature, measure, filters

from tfginfo.utils import Array, Image, create_bounding_box


def get_center_from_contour(contour: Array, shape: Tuple[int, int]) -> Array:
    # TODO optimize to bbox only
    cimg = np.zeros(shape)
    #cimg[contour.astype(np.uint32)] = 1
    rr, cc = draw.polygon(contour.T[0], contour.T[1], cimg.shape)
    cimg[rr, cc] = 1
    m = measure.moments(cimg)

    if m[0, 0] < 1e-4:
        raise ValueError("The countour is too small")
    return np.array([m[1, 0] / m[0, 0], m[0, 1] / m[0, 0]])


def create_line_iterator(p1: Array, p2: Array, img: Array) -> Tuple[Array, Array]:
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    :param p1: The first point.
    :param p2: The second point.
    :param img: The image being processed.

    :return: Coordinates and intensities of each pixel.

    """
    # define local variables for readability
    image_h = img.shape[0]
    image_w = img.shape[1]
    p1_x = p1[0]
    p1_y = p1[1]
    p2_x = p2[0]
    p2_y = p2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    d_x = p2_x - p1_x
    d_y = p2_y - p1_y
    d_xa = np.abs(d_x).astype(np.int64)
    d_ya = np.abs(d_y).astype(np.int64)

    # predefine numpy array for output based on distance between points
    it_buffer = np.empty(shape=(np.maximum(d_ya, d_xa), 3), dtype=np.float32)
    it_buffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    neg_y = p1_y > p2_y
    neg_x = p1_x > p2_x
    if p1_x == p2_x:  # vertical line segment
        it_buffer[:, 0] = p1_x
        if neg_y:
            it_buffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
        else:
            it_buffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
    elif p1_y == p2_y:  # horizontal line segment
        it_buffer[:, 1] = p1_y
        if neg_x:
            it_buffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
        else:
            it_buffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
    else:  # diagonal line segment
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

    # Remove points outside of image
    col_x = it_buffer[:, 0]
    col_y = it_buffer[:, 1]
    it_buffer = it_buffer[(col_x >= 0) & (col_y >= 0) & (col_x < image_w) & (col_y < image_h)]

    # Get intensities from img ndarray
    line_color = img[it_buffer[:, 1].astype(np.uint), it_buffer[:, 0].astype(np.uint)]

    return it_buffer[:, 0:2].astype(np.uint64), line_color.astype(np.uint64)


def get_corners_from_contour(contour: Array,
                             center: Array,
                             shape: Tuple[int, int],
                             num_corners: int,
                             min_distance: int) -> Array:
    # TODO optimize to bbox only
    rimg = np.zeros((shape[0], shape[1]))
    rr, cc = draw.polygon(contour.T[0], contour.T[1], rimg.shape)
    rimg[rr, cc] = 1

    # label_img = measure.label(rimg)
    # regions = measure.regionprops(label_img)
    # print(len(regions))
    # r = regions[0]
    #
    # rimg = np.zeros((shape[0], shape[1]))
    # rimg[r.convex_image] = 1

    # rimg = filters.gaussian(rimg, sigma=3)
    corners_measure = feature.corner_harris(rimg, sigma=min_distance / 15)#, sigma=20)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.imshow(rimg)
    # plt.figure()
    # plt.imshow(corners_measure)
    # plt.show()

    kwargs = dict(num_peaks=num_corners, min_distance=min_distance // 3)
    corners = feature.peak_local_max(corners_measure, **kwargs)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(r)
    # #plt.imshow(r.image)
    # #plt.imshow(r.convex_image)
    # # plt.show()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(rimg)
    # plt.scatter(*corners[:, ::-1].T)
    # plt.show()

    for i, c in enumerate(corners):
        vert = c[0] - center[0] > 1
        if vert:
            m = (c[1] - center[1]) / (c[0] - center[0])
            n = c[1] - m * c[0]

            if c[0] < center[0]:
                new_x = c[0] - min_distance
            else:
                new_x = c[0] + min_distance
            new_y = m * new_x + n
        else:
            m = (c[0] - center[0]) / (c[1] - center[1])
            n = c[0] - m * c[1]

            if c[1] < center[1]:
                new_y = c[1] - min_distance
            else:
                new_y = c[1] + min_distance
            new_x = m * new_y + n

        coords, vals = create_line_iterator(c[::-1], np.array([new_x, new_y])[::-1], rimg)
        index = np.nonzero(vals == 0)[0][0]
        corners[i] = coords[index - 1][::-1]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(rimg)
    # plt.scatter(*coords[vals == 0].T)
    # plt.scatter(*coords[vals == 1].T)
    # plt.scatter(*corners[:, ::-1].T)
    # plt.show()

    return corners


def get_ratios_from_center(image: Image, center: Array) -> Tuple[Array, Array]:
    pass
    # TODO


def get_contours_with_center(image: Image, center: Array, hratios: Array, vratios: Array):
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

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.gray()
    # plt.imshow(fimg)
    # for _, c, ce in candidates:
    #     plt.scatter(*ce[::-1].T)
    #     plt.plot(*c[:, ::-1].T)
    # plt.show()

    return candidates


def get_contour_from_center_and_ratios(image: Image, center: Array, hratios: Array,
                                       vratios: Array) -> Array:
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

        if is_finder_contour:# and is_exterior_contour:
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

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(fimg)
    # plt.plot(*contour[:, ::-1].T)
    # plt.show()

    # if len(candidates) > 1:
    #     raise ValueError("More than one contour found")
    # if len(candidates) == 0:
    #     raise ValueError("No contour found")
    # contour = candidates[0]

    if contour is None:
        raise ValueError("No contour found")

    contour += fimg_to_image
    # TODO maybe centroid is better center?
    return contour
