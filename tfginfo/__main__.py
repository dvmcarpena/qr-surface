from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pyzbar.pyzbar as zbar
from skimage import color, filters, feature, measure, draw
from tqdm import tqdm

from tfginfo.features import FinderPattern, find_finder_patterns, find_alignment_patterns
from tfginfo.utils import get_num_aligns_from_version


def check_ratio(arr):
    totalFinderSize = sum(arr[1:6])
    if totalFinderSize < 7:
        return False

    moduleSize = totalFinderSize // 7
    maxVariance = moduleSize // 2

    return (
        abs(moduleSize - arr[1]) < maxVariance
        and abs(moduleSize - arr[2]) < maxVariance
        and abs(3 * moduleSize - arr[3]) < 3 * maxVariance
        and abs(moduleSize - arr[4]) < maxVariance
        and abs(moduleSize - arr[5]) < maxVariance
    )


def centerFromEnd(arr, beg):
    middle = len(arr) // 2
    return beg + sum(arr[:middle]) + arr[middle] // 2


def centerFromEnd2(arr, beg):
    middle = len(arr) // 2
    return beg + sum(arr[:middle]) + arr[middle] // 2


def crossCheckVertical(data, row, pr=False):
    magic_num_1 = 4

    changes_v = []
    changes = []
    last = None
    rep = 0
    k = 0
    while len(data) > row + k and len(changes) <= magic_num_1:
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
    while row - k >= 0 and len(changesd) <= magic_num_1:
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

    if len(changes) == magic_num_1 + 1 and len(changesd) == magic_num_1 + 1:
        th = list(reversed(changesd))[1:-1] + [changesd[0] + changes[0]] + changes[1:-1]
        if check_ratio(th):
            return centerFromEnd2(th[1:-1], changesd_v[-2][1])

    return None

def handlePossibleCenter(img, arr, row, centerCol):
    centerRow = crossCheckVertical(img[:, centerCol], row)
    if centerRow is None:
        return None

    centerCol = crossCheckVertical(img[centerRow, :], centerCol)
    if centerCol is None:
        return None

    # TODO diag check

    return centerRow, centerCol


def find_finder_patterns2(img: np.ndarray, block_size=151, offset=0):
    img_gray: np.ndarray = color.rgb2gray(img)
    #binary_adaptive = filters.threshold_local(img_gray, block_size, offset=offset)
    binary_adaptive: np.ndarray = filters.threshold_sauvola(img_gray, block_size)
    #binary_adaptive = filters.threshold_otsu(img_gray)
    #fig, ax = filters.try_all_threshold(img_gray, figsize=(15, 20), verbose=True)
    #fig.savefig("aaaa.png")
    bin_img: np.ndarray = img_gray > binary_adaptive
    eps = 0
    ratios = []
    candidates = []
    final_centers = []
    for i, line in filter(lambda x: x[0] % 4 == 0, #and x[0] == 1252,
                          enumerate(tqdm(bin_img, ncols=100))):
        line_val = []
        line_rep = []
        last = None
        rep = 0
        for j, first in enumerate(line):
            if last is None:
                last = first
                line_val.append([first, j])
                line_rep.append(1)

            if last == first:
                line_rep[rep] += 1
            else:
                last = first
                line_val.append([first, j])
                line_rep.append(1)
                rep += 1

        it = filter(
            lambda x: x[0] < len(line_rep) - 6 and x[1][0],
            enumerate(line_val)
        )
        for k, (_, idx) in it:
            if check_ratio(line_rep[k:k + 7]):
                candidates.append([i, idx, line_rep[k:k + 7]])

    centerss = []
    for row, col, arr in candidates:
        centerCol = centerFromEnd(arr, col)
        size = sum(arr[1:6])

        if any(abs(centerCol - finder.center[1]) < size and abs(row - finder.center[0]) < size
               for finder in final_centers):
            continue

        center = handlePossibleCenter(bin_img, arr, row, centerCol)
        if center is not None:
            #centerss.append([center[0], center[1], arr])
            center = np.array([center[0], center[1]])
            final_centers.append(FinderPattern.from_center_and_ratios(
                image=bin_img,
                center=center,
                hratios=arr[1:6],
                vratios=arr[1:6]
            ))

    # for c, r, arr in centerss:
    #     center = np.array([c, r])
    #     finder_pattern = FinderPattern.from_center_and_ratios(bin_img, center, arr[1:6], arr[1:6])
    #     final_centers.append(finder_pattern)
        # size = sum(arr[1:6])
        # cut_s = int(size * 0.75)
        # small_rad = sum(arr[2:5]) // 2
        # c_img = bin_img[c - cut_s:c + cut_s, r - cut_s:r + cut_s]
        # contours = measure.find_contours(c_img, 0.8)
        #
        # c_orig = c
        # r_orig = r
        # c = cut_s
        # r = cut_s
        #
        # for n, contour in enumerate(contours):
        #     mm = np.zeros_like(c_img)
        #     mm[contour.astype(np.uint32)] = 1
        #     M = measure.moments(mm)
        #     centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
        #     test_point = [[c, r - arr[2] - arr[3] // 2 - arr[1] // 2]]
        #     if abs(centroid[0] - c) < small_rad and abs(centroid[1] - r) < small_rad and measure.points_in_poly(test_point, contour)[0]:
        #         cnt = contour + np.array([c_orig - cut_s, r_orig - cut_s])
        #         centroid = np.array(centroid) + np.array([c_orig - cut_s, r_orig - cut_s])
        #         #test_point = np.array(test_point[0]) + np.array([c_orig - cut_s, r_orig - cut_s])
        #         cors = np.zeros_like(c_img)
        #         rr, cc = draw.polygon(contour.T[0], contour.T[1], cors.shape)
        #         cors[rr, cc] = 1
        #         coords = feature.corner_peaks(feature.corner_harris(cors, sigma=4),
        #                                       num_peaks=4, min_distance=size // 5)
        #
        #         final_centers.append([c, r, arr, centroid, cnt, coords])
        #         print("byba")

    return bin_img,  np.array(candidates)[:, :-1], final_centers


def find_finder_patterns_zbar(img: np.ndarray):
    print(zbar.decode(np.array(img)))


def find_finder_patterns_ocv(img: np.ndarray):
    img = img[:, :, ::-1]
    detector = cv2.QRCodeDetector()
    detector.setEpsX(.1)
    detector.setEpsY(.1)
    found, bbox = detector.detect(img)
    data, *_ = detector.detectAndDecode(img)
    print("aaa" + data)
    if not found:
        raise ValueError("Not found any QR Code")

    return bbox


if __name__ == "__main__":
    image_paths = [
        ("../test.png", 7),
        ("../imgs/IMG_20191225_202803.jpg", 7),
        ("../imgs/IMG_20191225_202839.jpg", 7),
        ("../imgs/2019-11-07_11-09-38_117646702.png", 3)
    ]

    # plt.figure()
    # plt.gray()
    # plt.imshow(feature.canny(color.rgb2gray(image)[800:1000, 1250:1450], sigma=3))
    # plt.show()
    # img_l = feature.canny(color.rgb2gray(image)[800:1000, 1250:1450], sigma=3)
    # thr = filters.threshold_otsu(color.rgb2gray(image)[800:1000, 1250:1450])
    # img_l2 = color.rgb2gray(image)[800:1000, 1250:1450] > thr
    # # Find contours at a constant value of 0.8
    # contours = measure.find_contours(img_l2, 0.8)
    #
    # # Display the image and plot all contours found
    # fig, ax = plt.subplots()
    # ax.imshow(img_l2, cmap=plt.cm.gray)
    #
    # for n, contour in enumerate(contours):
    #     mm = np.zeros_like(img_l2)
    #     for c in contour:
    #         mm[int(c[0]), int(c[1])] = 1
    #
    #     M = measure.moments(mm)
    #     centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    #     if centroid[0] < 150 and centroid[1] < 150 and measure.points_in_poly([[83, 27]], contour):
    #         ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    #         ax.scatter(centroid[1], centroid[0])
    #
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    #img_gray: np.ndarray = color.rgb2gray(img)
    #binary_adaptive = filters.threshold_local(img_gray, block_size, offset=offset)
    #binary_adaptive: np.ndarray = filters.threshold_sauvola(img_gray, block_size)
    #binary_adaptive = filters.threshold_otsu(img_gray)
    #fig, ax = filters.try_all_threshold(img_gray, figsize=(15, 20), verbose=True)
    #fig.savefig("aaaa.png")

    #result = find_finder_patterns_ocv(image)
    #find_finder_patterns_zbar(image)
    #print(result)
    #plt.figure()
    #plt.imshow(image)
    #plt.scatter(*result.T)
    #img_bin, candidates, fc = find_finder_patterns2(image)
    for image_path, version in image_paths:
        name = Path(image_path).name
        image = imageio.imread(image_path)
        num_aligns = get_num_aligns_from_version(version)

        fc = find_finder_patterns(image)
        ap = find_alignment_patterns(image)

        if len(fc) != 3:
            print(f"{name}: Finder patterns error")
        if len(ap) != num_aligns:
            print(f"{name}: Alignment patterns error")

        centroid = np.array([finder.center for finder in fc])
        contours = np.array([finder.contour for finder in fc])
        corns = np.array([finder.corners for finder in fc])

        ap_centers = np.array([align.center for align in ap])
        ap_contour = np.array([align.contour for align in ap if align.contour is not None])
        ap_corners = np.array([align.corners for align in ap if align.corners is not None])

        plt.figure()
        plt.title(name)
        plt.imshow(image)
        plt.scatter(*centroid[:, ::-1].T)
        plt.scatter(*corns[:, ::-1].T)
        plt.scatter(*ap_centers[:, ::-1].T)
        plt.scatter(*ap_corners[:, ::-1].T)

        for cnt in contours:
            plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
        for cnt in ap_contour:
           plt.plot(cnt[:, 1], cnt[:, 0], linewidth=2)

    plt.show()
