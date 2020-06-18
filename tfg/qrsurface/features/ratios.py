import itertools
from typing import Tuple, List, Optional, Union

import numpy as np

from .utils import get_contours_with_center


def center_from_end(arr: Union[np.ndarray, List], beg: int) -> int:
    """
    Get the center of a pattern given an array of ratios and the beginning pixel

    :param arr: Array of ratios
    :param beg: The beginning pixel

    :return: Middle pixel
    """
    middle = len(arr) // 2
    return beg + sum(arr[:middle]) + arr[middle] // 2


def check_general_ratio(ratios: List, widths: List, strict_border: bool) -> bool:
    """
    Checks if the widths given follow the ratios

    :param ratios: List of ratios
    :param widths: Widths of the sequences of pixels
    :param strict_border: Whether to use a strict border

    :return: Whether the given widths follow the ratios given
    """
    if strict_border:
        size = sum(widths)
        lenght = sum(ratios)
    else:
        size = sum(widths[1:-1])
        lenght = sum(ratios[1:-1])

    if size < lenght:
        return False

    module_size = size // lenght
    max_variance = module_size / 2

    if strict_border:
        middle = zip(ratios, widths)
        border_ratios = []
    else:
        middle = zip(ratios[1:-1], widths[1:-1])
        border_ratios = [
            widths[0] - ratios[0] * module_size > - ratios[0] * max_variance,
            widths[-1] - ratios[-1] * module_size > - ratios[-1] * max_variance
        ]

    middle_ratios = [
        abs(ratio * module_size - width) < ratio * max_variance
        for ratio, width in middle
    ]

    return all(middle_ratios + border_ratios)


def compute_widths_by_row(row: np.ndarray) -> Tuple[List, List]:
    """
    Given a row, computes the sequences of successive pixels

    :param row: A array of binarized colors

    :return: An array of the values of the sequences and the widths of the sequences
    """
    nline_rep = [sum(1 for _ in group) for key, group in itertools.groupby(row)]
    nline_val: np.ndarray
    if not row[0]:
        nline_val = np.ones_like(nline_rep)
        nline_val[::2] = 0.
    else:
        nline_val = np.zeros_like(nline_rep)
        nline_val[::2] = 1.

    # noinspection PyTypeChecker
    line_val: List = nline_val.tolist()
    acc = 0
    for i, rep in enumerate(nline_rep):
        line_val[i] = [bool(line_val[i]), acc]
        acc += rep

    return line_val, nline_rep


def cross_check(data: np.ndarray, row: int, iratios: List, strict_border: bool) -> Optional[int]:
    """
    Checks if the found center complies with the ratios in different orientations

    :param data: The target orientation array to check the ratios
    :param row: The current row number
    :param iratios: The array os ratios to search for
    :param strict_border: Whether the pattern has strict border

    :return: The found column for the center
    """
    len_side = len(iratios) // 2 + 2

    changes_v = []
    changes = []
    last = None
    rep = 0
    k = 0
    while len(data) > row + k and len(changes) < len_side:
        first = data[row + k]
        if last is None:
            last = first
            changes_v.append([first, row + k])
            changes.append(1)

        if last == first:
            changes[rep] += 1
        else:
            last = first
            changes_v.append([first, row + k])
            changes.append(1)
            rep += 1

        k += 1
    if len(data) <= row + k:
        changes_v.append([not last, row + k])
        changes.append(1)

    changesd_v = []
    changesd = []
    last = None
    rep = 0
    k = 1
    while row - k >= 0 and len(changesd) < len_side:
        first = data[row - k]
        if last is None:
            last = first
            changesd_v.append([first, row - k])
            changesd.append(1)

        if last == first:
            changesd[rep] += 1
        else:
            last = first
            changesd_v.append([first, row - k])
            changesd.append(1)
            rep += 1

        k += 1
    if row - k < 0:
        changesd_v.append([not first, row - k])
        changesd.append(1)

    if len(changes) != len_side or len(changesd) != len_side:
        return None

    th = list(reversed(changesd))[1:-1] + [changesd[0] + changes[0]] + changes[1:-1]

    if not check_general_ratio(iratios, th, strict_border=strict_border):
        return None

    if strict_border:
        widths = th
        new_center = center_from_end(widths, changesd_v[-1][1])
    else:
        widths = th[1:-1]
        new_center = center_from_end(widths, changesd_v[-2][1])

    return new_center


def handle_possible_center(img: np.ndarray, row: int, center_col: int, iratios: List, strict_border: bool,
                           diagonals: bool = True) -> Optional[Tuple[int, int]]:
    """
    Test whether a possible center is a valid pattern

    :param img: The image where we are searching for patterns
    :param row: The row where the center was found
    :param center_col: The column that indicates what pixel was the possible center
    :param iratios: The array os ratios to search for
    :param strict_border: Whether the pattern has strict border
    :param diagonals: Whether to check the diagonals

    :return: The coordinates of the center or None
    """
    center_row = cross_check(img[:, center_col], row, iratios, strict_border)
    if center_row is None:
        return None

    center_col = cross_check(img[center_row, :], center_col, iratios, strict_border)
    if center_col is None:
        return None

    if diagonals:
        d = max(center_row, center_col) - min(center_row, center_col)
        if center_row >= center_col:
            diag1_start = d
            diag2_start = 0
            f = min(img.shape[0] - 1 - d, img.shape[1] - 1)
            if img.shape[0] - d >= img.shape[1]:
                diag1_end = d + f
                diag2_end = img.shape[1] - 1
            else:
                diag1_end = img.shape[0] - 1
                diag2_end = f
        else:
            diag1_start = 0
            diag2_start = d
            f = min(img.shape[1] - 1 - d, img.shape[0] - 1)
            if img.shape[1] - d >= img.shape[0]:
                diag1_end = img.shape[0] - 1
                diag2_end = d + f
            else:
                diag1_end = f
                diag2_end = img.shape[1] - 1
        diag1 = list(range(diag1_start, diag1_end))
        diag2 = list(range(diag2_start, diag2_end))

        index = max(center_row, center_col) - d
        check = cross_check(img[diag1, diag2], index, iratios, strict_border)
        if check is None:
            return None

        inv_col = img.shape[1] - 1 - center_col
        if center_row >= inv_col:
            diag1_inv_start = center_row - inv_col
            diag2_inv_start = img.shape[1] - 1
            if img.shape[1] + center_row - inv_col >= img.shape[0]:
                diag1_inv_end = img.shape[0] - 1
                diag2_inv_end = img.shape[1] - 1 - ((img.shape[0] - 1) - (center_row - inv_col))
            else:
                diag1_inv_end = img.shape[1] - 1 + center_row - inv_col
                diag2_inv_end = 0
        else:
            diag1_inv_start = 0
            diag2_inv_start = center_col + center_row
            if center_col + center_row >= img.shape[0]:
                diag1_inv_end = img.shape[0] - 1
                diag2_inv_end = center_col + center_row - (img.shape[0] - 1)
            else:
                diag1_inv_end = center_col + center_row
                diag2_inv_end = 0
        diag1_inv = list(range(diag1_inv_start, diag1_inv_end + 1))
        diag2_inv = list(range(diag2_inv_start, diag2_inv_end - 1, -1))

        index = min(inv_col, center_row)
        check = cross_check(img[diag1_inv, diag2_inv], index, iratios, strict_border)
        if check is None:
            return None

    return center_row, center_col


def check_contours(bw_image: np.ndarray, center_row: int, center_col: int, hratios: List,
                   strict_border: bool) -> Optional[Tuple[int, int]]:
    """
    Checks if the found center has the correct number of contours around

    :param bw_image: The binary image where to search the patterns
    :param center_row: The found center row
    :param center_col: The found center column
    :param hratios: The array of found ratios
    :param strict_border: Whether the pattern has strict border

    :return: The coordinates of the center or None
    """
    cands = get_contours_with_center(bw_image, np.array([center_row, center_col]), hratios, hratios)

    if strict_border:
        expected_length = len(hratios) // 2 + 1
    else:
        expected_length = len(hratios) // 2

    if len(cands) < expected_length:
        return None

    return center_row, center_col


def find_general(bw_image: np.ndarray, iratios: List, center_color: bool, strict_border: bool, diagonals: bool,
                 countours: bool) -> List[List[np.ndarray]]:
    """
    General method for finding patterns using ratio-based approach

    :param bw_image: The binary image where to search the patterns
    :param iratios: The array of target ratios
    :param center_color: The color of the center pixel
    :param strict_border: Whether the pattern has strict border
    :param diagonals: Whether to check the diagonals
    :param countours: Whether to check the contour

    :return: List of found patterns, given by a center and pair of ratios
    """
    border_color = bool((len(iratios) // 2 + int(center_color)) % 2)

    candidates = []
    for i, row in filter(lambda x: x[0] % 4 == 0, enumerate(bw_image)):
        line_val, line_rep = compute_widths_by_row(row)

        it = filter(
            lambda x: x[1][0] == border_color,
            enumerate(line_val)
        )
        for k, (_, idx) in it:
            widths = line_rep[k:k + len(iratios)]

            if k + len(widths) - 1 < len(line_rep) \
                    and check_general_ratio(iratios, widths, strict_border):
                candidates.append([i, idx, widths])

    final_centers = []
    for row, col, widths in candidates:

        if strict_border:
            hratios = widths
        else:
            hratios = widths[1:-1]
        center_col = center_from_end(widths, col)
        size = sum(hratios)

        if any(abs(center_col - c[1]) < size and abs(row - c[0]) < size
               for c, _, _ in final_centers):
            continue

        center = handle_possible_center(bw_image, row, center_col, iratios, strict_border, diagonals=diagonals)
        if countours:
            center = (check_contours(bw_image, center[0], center[1], hratios, strict_border)
                      if center is not None else None)
        if center is not None:
            center = np.array([center[0], center[1]])
            final_centers.append([center, hratios, hratios])

    return final_centers
