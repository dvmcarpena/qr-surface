from typing import Tuple, List, Optional

import numpy as np
from tqdm import tqdm

from tfginfo.utils import Array, Image

from .utils import get_contours_with_center


def centerFromEnd(arr, beg):
    middle = len(arr) // 2
    return beg + sum(arr[:middle]) + arr[middle] // 2


def centerFromEnd2(arr, beg):
    middle = len(arr) // 2
    return beg + sum(arr[:middle]) + arr[middle] // 2


def check_general_ratio(ratios, widths, strict_border: bool) -> bool:
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


def compute_widths_by_row(row: Array) -> Tuple[List, List]:
    # line_val = []
    # line_rep = []
    # last = None
    # rep = 0
    # for j, first in enumerate(row):
    #     if last is None:
    #         last = first
    #         line_val.append([first, j])
    #         line_rep.append(1)
    #     elif last == first:
    #         line_rep[rep] += 1
    #     else:
    #         last = first
    #         line_val.append([first, j])
    #         line_rep.append(1)
    #         rep += 1

    # nline_rep = np.diff(np.where(np.concatenate(([row[0]],
    #                                              row[:-1] != row[1:],
    #                                              [True])))[0])[::2]
    import itertools
    nline_rep = [sum(1 for _ in group) for key, group in itertools.groupby(row)]
    if not row[0]:
        nline_val = np.ones_like(nline_rep)
        nline_val[::2] = 0.
    else:
        nline_val = np.zeros_like(nline_rep)
        nline_val[::2] = 1.

    nline_val = nline_val.tolist()
    acc = 0
    for i, rep in enumerate(nline_rep):
        nline_val[i] = [bool(nline_val[i]), acc]
        acc += rep

    # print(row[-30:].tolist())
    # print(nline_rep)
    # print(line_rep)
    # print(nline_val)
    # print(line_val)

    # return line_val, line_rep
    return nline_val, nline_rep


def cross_check(data, row, iratios, strict_border: bool) -> Optional[int]:
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
        new_center = centerFromEnd2(widths, changesd_v[-1][1])
    else:
        widths = th[1:-1]
        new_center = centerFromEnd2(widths, changesd_v[-2][1])

    return new_center


def handle_possible_center(img, row, centerCol, iratios, strict_border, diagonals=True):
    centerRow = cross_check(img[:, centerCol], row, iratios, strict_border)
    if centerRow is None:
        return None

    centerCol = cross_check(img[centerRow, :], centerCol, iratios, strict_border)
    if centerCol is None:
        return None

    if diagonals:
        d = max(centerRow, centerCol) - min(centerRow, centerCol)
        if centerRow >= centerCol:
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

        # plt.figure()
        # plt.imshow(img)
        # plt.scatter(centerCol, centerRow)
        # plt.plot(diag2[:min(len(diag1), len(diag2))], diag1[:min(len(diag1), len(diag2))])
        # plt.show()

        index = max(centerRow, centerCol) - d
        check = cross_check(img[diag1, diag2], index, iratios, strict_border)
        if check is None:
            return None

        inv_col = img.shape[1] - 1 - centerCol
        if centerRow >= inv_col:
            diag1_inv_start = centerRow - inv_col
            diag2_inv_start = img.shape[1] - 1
            if img.shape[1] + centerRow - inv_col >= img.shape[0]:
                diag1_inv_end = img.shape[0] - 1
                diag2_inv_end = img.shape[1] - 1 - ((img.shape[0] - 1) - (centerRow - inv_col))
            else:
                diag1_inv_end = img.shape[1] - 1 + centerRow - inv_col
                diag2_inv_end = 0
        else:
            diag1_inv_start = 0
            diag2_inv_start = centerCol + centerRow
            if centerCol + centerRow >= img.shape[0]:
                diag1_inv_end = img.shape[0] - 1
                diag2_inv_end = centerCol + centerRow - (img.shape[0] - 1)
            else:
                diag1_inv_end = centerCol + centerRow
                diag2_inv_end = 0
        diag1_inv = list(range(diag1_inv_start, diag1_inv_end + 1))
        diag2_inv = list(range(diag2_inv_start, diag2_inv_end - 1, -1))

        # plt.figure()
        # plt.imshow(img)
        # plt.scatter(centerCol, centerRow)
        # plt.plot(diag2, diag1)
        # plt.plot(diag2_inv[:min(len(diag2_inv), len(diag1_inv))], diag1_inv[:min(len(diag2_inv), len(diag1_inv))])
        # plt.show()

        index = min(inv_col, centerRow)
        check = cross_check(img[diag1_inv, diag2_inv], index, iratios, strict_border)
        if check is None:
           return None

    return centerRow, centerCol


def check_contours(bw_image, centerRow, centerCol, hratios, strict_border):
    cands = get_contours_with_center(bw_image, np.array([centerRow, centerCol]), hratios, hratios)

    # print(len(cands), len(hratios))
    if strict_border:
        l = len(hratios) // 2 + 1
    else:
        l = len(hratios) // 2

    if len(cands) < l:
        return None

    return centerRow, centerCol


def find_general(bw_image: Image, iratios, center_color: bool,
                 strict_border: bool, diagonals, countours) -> List[List[Array]]:
    border_color = bool((len(iratios) // 2 + int(center_color)) % 2)

    candidates = []
    for i, row in filter(lambda x: x[0] % 4 == 0, enumerate(bw_image)):
        line_val, line_rep = compute_widths_by_row(row)

        # fr = border_color == row[0]
        # print([val == border_color for val, _ in line_val])
        # print([(i % 2 == 0) == fr for i, _ in enumerate(line_val)])
        it = filter(
            lambda x: x[1][0] == border_color,
            # lambda x: (x[0] % 2 == 0) == fr,
            enumerate(line_val)
        )
        for k, (_, idx) in it:
            widths = line_rep[k:k + len(iratios)]

            if k + len(widths) - 1 < len(line_rep) \
                    and check_general_ratio(iratios, widths, strict_border):
                candidates.append([i, idx, widths])

    # import matplotlib.pyplot as plt
    # centers = np.array([[i, centerFromEnd(widths, idx)] for i, idx, widths in candidates])
    # plt.figure()
    # plt.gray()
    # plt.imshow(bw_image)
    # plt.scatter(*centers[:, ::-1].T)
    # plt.show()

    final_centers = []
    for row, col, widths in candidates:

        if strict_border:
            hratios = widths
        else:
            hratios = widths[1:-1]
        centerCol = centerFromEnd(widths, col)
        size = sum(hratios)

        if any(abs(centerCol - c[1]) < size and abs(row - c[0]) < size
               for c, _, _ in final_centers):
            continue

        center = handle_possible_center(bw_image, row, centerCol, iratios, strict_border, diagonals=diagonals)
        if countours:
            center = check_contours(bw_image, center[0], center[1], hratios, strict_border) if center is not None else None
        if center is not None:
            center = np.array([center[0], center[1]])
            final_centers.append([center, hratios, hratios])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.gray()
    # plt.imshow(bw_image)
    # plt.scatter(*np.array([c for c, _, _ in final_centers])[:, ::-1].T)
    # plt.show()

    return final_centers
