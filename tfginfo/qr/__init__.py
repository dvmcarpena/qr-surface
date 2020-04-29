import copy
from dataclasses import dataclass
from enum import auto, Enum, unique
import itertools
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance
from skimage import img_as_ubyte, color
from sklearn import cluster

from tfginfo.utils import (Array, Image, get_num_aligns_from_version, get_size_from_version,
                           get_alignments_centers, create_bounding_box)
from tfginfo.features import AlignmentPattern, Features, FinderPattern
from tfginfo.decode import decode
from tfginfo.matching import MatchingFeatures, References

from .utils import guess_version

OrderedFinderPatterns = Tuple[FinderPattern, FinderPattern, FinderPattern]


def group_finder_patterns(finder_patterns: List[FinderPattern]) -> Iterator[List[FinderPattern]]:
    if len(finder_patterns) < 3:
        return
    if len(finder_patterns) == 3:
        yield finder_patterns
        return

    finder_patterns = np.array(finder_patterns)
    centers = np.array([f.center for f in finder_patterns])

    num_qrcodes = len(finder_patterns) // 3
    kmeans = cluster.KMeans(n_clusters=num_qrcodes, random_state=0)
    kmeans.fit(centers)
    predict = kmeans.predict(centers)
    dists = kmeans.transform(centers)

    for value in np.unique(np.array(predict)):
        f = finder_patterns[predict == value]
        d = dists.T[value][predict == value]
        if len(f) == 3:
            yield f.tolist()
            continue
        if len(f) < 3:
            continue

        indexes = np.argsort(d)
        yield f[indexes[:3]].tolist()


def orientate_finder_patterns(finder_patterns: List[FinderPattern]) -> OrderedFinderPatterns:
    assert len(finder_patterns) == 3

    centers = [f.center for f in finder_patterns]
    centers_indexes = [0, 1, 2]

    # We can group in pairs the candidates in order to compute their relative distance.
    centers_pairs = list(itertools.combinations(centers, 2))
    centers_dists = [distance.euclidean(c1, c2) for c1, c2 in centers_pairs]

    # Then we create all the possible permutations of these distances and check one that majorizes.
    major_dist_map = list(map(lambda x: (x[0] > x[1]) and (x[0] > x[2]), itertools.permutations(centers_dists)))
    index = major_dist_map.index(True)
    index_2 = list(itertools.permutations([0, 1, 2]))[index][0]
    diagonal_centers = centers_pairs[index_2]

    pair_diag = list(itertools.combinations(centers_indexes, 2))[index_2]
    f2, f3 = finder_patterns[pair_diag[0]], finder_patterns[pair_diag[1]]
    centers_indexes.remove(pair_diag[0])
    centers_indexes.remove(pair_diag[1])
    first_center = centers[centers_indexes[0]]
    f1 = finder_patterns[centers_indexes[0]]

    ab = diagonal_centers[0] - first_center
    bc = diagonal_centers[1] - first_center
    # minor = np.linalg.det(np.stack(((ab / np.linalg.norm(ab))[-2:],
    #                                 (bc / np.linalg.norm(bc))[-2:])))

    if np.sign(np.linalg.det(np.stack((ab, bc)))) < 0:
        fins = f1, f2, f3
    else:
        fins = f1, f3, f2

    # Reorder corners
    f1, f2, f3 = fins
    v1 = f2.center - f1.center
    v2 = f3.center - f1.center
    det = v1[0] * v2[1] - v1[1] * v2[0]
    mat = np.array([
        [v2[1] / det, - v2[0] / det, 0],
        [- v1[1] / det, v1[0] / det, 0],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, - f1.center[0]],
        [0, 1, - f1.center[1]],
        [0, 0, 1]
    ])

    new_cors = [None, None, None, None]
    for c in f1.corners:
        new_cor = (mat @ np.array([
            [c[0]],
            [c[1]],
            [1]
        ])).T[0]
        c = c.tolist()

        if new_cor[0] < 0  and new_cor[1] < 0:
            new_cors[0] = c
        elif new_cor[0] > 0  and new_cor[1] < 0:
            new_cors[1] = c
        elif new_cor[0] > 0  and new_cor[1] > 0:
            new_cors[2] = c
        else:# new_cor[0] < 0  and new_cor[1] > 0:
            new_cors[3] = c
    f1.corners = np.array(new_cors)

    new_cors = [None, None, None, None]
    for c in f2.corners:
        new_cor = (mat @ np.array([
            [c[0]],
            [c[1]],
            [1]
        ])).T[0]
        c = c.tolist()

        if new_cor[1] < 0:
            if new_cor[0] < 1:
                new_cors[0] = c
            else:
                new_cors[1] = c
        else:
            if new_cor[0] > 1:
                new_cors[2] = c
            else:
                new_cors[3] = c
    f2.corners = np.array(new_cors)

    new_cors = [None, None, None, None]
    for c in f3.corners:
        new_cor = (mat @ np.array([
            [c[0]],
            [c[1]],
            [1]
        ])).T[0]
        c = c.tolist()

        if new_cor[1] < 1:
            if new_cor[0] < 0:
                new_cors[0] = c
            else:
                new_cors[1] = c
        else:
            if new_cor[0] > 0:
                new_cors[2] = c
            else:
                new_cors[3] = c
    f3.corners = np.array(new_cors)

    return fins


def guess_version_from_finders(finder_patterns: OrderedFinderPatterns) -> int:
    version_points = np.array([
        finder_patterns[0].corners[0].tolist(),
        finder_patterns[0].corners[3].tolist(),
        finder_patterns[2].corners[0].tolist(),
        finder_patterns[2].corners[3].tolist(),
    ])
    return guess_version(version_points)


def choose_and_order_alignments(finder_patterns: OrderedFinderPatterns,
                                version: int,
                                alignment_patterns: List[AlignmentPattern]
                                ) -> List[AlignmentPattern]:
    final_ap = []

    f1, f2, f3 = finder_patterns
    # v1 = f2.center - f1.center
    # v2 = f3.center - f1.center
    v1 = f2.corners[1] - f1.corners[0]
    v2 = f3.corners[3] - f1.corners[0]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    mat = np.array([
        [v2[1] / det, - v2[0] / det, 0],
        [- v1[1] / det, v1[0] / det, 0],
        [0, 0, 1]
    ]) @ np.array([
        # [1, 0, - f1.center[0]],
        # [0, 1, - f1.center[1]],
        [1, 0, - f1.corners[0][0]],
        [0, 1, - f1.corners[0][1]],
        [0, 0, 1]
    ])
    from skimage import transform

    eps = 1e-4

    x11 = f2.corners[1][0]
    y11 = f2.corners[1][1]
    x12 = f2.corners[2][0]
    y12 = f2.corners[2][1]

    # a1 = (y11 - y12) / (x11 - x12)
    # b1 = y11 - a1 * x11
    if abs(x11 - x12) <= eps:
        a1 = 1
        b1 = 0
    else:
        a1 = - (y11 - y12) / (x11 - x12)
        b1 = 1
    c1 = a1 * x11 + b1 * y11

    x21 = f3.corners[3][0]
    y21 = f3.corners[3][1]
    x22 = f3.corners[2][0]
    y22 = f3.corners[2][1]

    # a2 = (y21 - y22) / (x21 - x22)
    # b2 = y21 - a2 * x21
    if abs(x21 - x22) <= eps:
        a2 = 1
        b2 = 0
    else:
        a2 = - (y21 - y22) / (x21 - x22)
        b2 = 1
    c2 = a2 * x21 + b2 * y21

    # x = (b2 - b1) / (a1 - a2)
    # y = a1 * x + b1
    fourth_corner = np.linalg.solve(np.array([[a1, b1], [a2, b2]]), np.array([c1, c2]))

    _src = np.array([
        f1.corners[0].tolist(),
        f2.corners[1].tolist(),
        f3.corners[3].tolist(),
        fourth_corner.tolist()
    ])
    _dst = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    _m = transform.ProjectiveTransform()
    _m.estimate(_src, _dst)

    # limits_qr = (-0.25, 1.25)
    # limits_qr = (-0.1, 1.1)
    limits_qr = (0, 1)
    for ap in alignment_patterns:
        # new_ap = (mat @ np.array([
        #     [ap.center[0]],
        #     [ap.center[1]],
        #     [1]
        # ])).T[0]
        new_ap = _m(ap.center)[0]

        if limits_qr[0] < new_ap[0] < limits_qr[1] and limits_qr[0] < new_ap[1] < limits_qr[1]:
            final_ap.append(ap)

    # if len(final_ap) != get_num_aligns_from_version(version):
    #     warnings.warn(f"Found {len(final_ap)} alignment patterns, but with the "
    #                   f"estimed version {version} it should have "
    #                   f"{get_num_aligns_from_version(version)}.")

    ideal_positions = get_alignments_centers(version)
    total_size = get_size_from_version(version)
    ideal_positions = np.array(ideal_positions) / total_size
    ideal_positions = ideal_positions[np.lexsort((ideal_positions[:, 0], ideal_positions[:, 1]))]

    def transf(ap: AlignmentPattern) -> Tuple[AlignmentPattern, Array]:
        # return ap, (mat @ np.array([
        #     [ap.center[0]],
        #     [ap.center[1]],
        #     [1]
        # ])).T[0][:2]
        return ap, _m(ap.center)[0]

    # print(np.array([c for ap, c in map(transf, final_ap)]))
    # print(ideal_positions)
    # from scipy import spatial
    # dists = spatial.distance_matrix(
    #     np.array([c for ap, c in map(transf, final_ap)]),
    #     ideal_positions
    # )
    # selection = np.argmin(dists, axis=0)
    # selection2 = np.argmin(dists, axis=1)

    # def transfm1(point):
    #     # return (np.linalg.inv(mat) @ np.array([
    #     #     [point[0]],
    #     #     [point[1]],
    #     #     [1]
    #     # ])).T[0][:2]
    #     return _m.inverse(point)[0]

    # # print(dists)
    # print(selection)
    # print(selection2)
    # # print(np.unique(selection, return_inverse=True, return_counts=True))
    # if len(np.unique(selection)) < min(get_num_aligns_from_version(version)):
    #     np.unique(selection, return_counts=True)
    #
    # print(np.array([c for ap, c in map(transf, final_ap)]))
    # print(ideal_positions)
    # print(final_ap)
    # ordered_ap = np.array(final_ap)[selection].tolist()
    # print(ordered_ap)

    # cor1 = transfm1([0, 0])
    # cor2 = transfm1([1, 0])
    # cor3 = transfm1([0, 1])
    # cor4 = transfm1([1, 1])

    # plt.scatter(fourth_corner[1], fourth_corner[0])
    # plt.scatter([cor1[1], cor2[1], cor3[1], cor4[1]],
    #             [cor1[0], cor2[0], cor3[0], cor4[0]])
    # plt.scatter([f1.corners[0][1], f2.corners[1][1], f3.corners[3][1]],
    #             [f1.corners[0][0], f2.corners[1][0], f3.corners[3][0]])

    ordered_ap = []
    # rad = 1.5 * ideal_positions[0][1]
    rad = 0.5 * ideal_positions[0][1]
    for ideal_align in ideal_positions:
        margins = (
            (ideal_align[0] - rad, ideal_align[1] - rad),
            (ideal_align[0] + rad, ideal_align[1] + rad)
        )
        # _p1 = transfm1(margins[0])
        # _p2 = transfm1(margins[1])
        # i = transfm1(ideal_align)

        # plt.scatter([_p1[1], _p2[1]], [_p1[0], _p2[0]])
        # plt.scatter(i[1], i[0])

        candidates = [
            (ap, c)
            for ap, c in map(transf, final_ap)
            if margins[0][0] < c[0] < margins[1][0] and margins[0][1] < c[1] < margins[1][1]
        ]

        if len(candidates) == 0:
            ordered_ap.append(None)
        elif len(candidates) == 1:
            ordered_ap.append(candidates[0][0])
        else:
            index = int(np.argmin([distance.euclidean(ideal_align, cand[1]) for cand in candidates]))
            ordered_ap.append(candidates[index][0])

    return ordered_ap


@unique
class Correction(Enum):
    AFFINE = auto()
    PROJECTIVE = auto()
    CYLINDRICAL = auto()
    TPS = auto()


@dataclass
class QRCode:
    image: Image
    finder_patterns: OrderedFinderPatterns
    version: int
    alignment_patterns: List[Optional[AlignmentPattern]]
    fourth_corner: Optional[Array]

    @classmethod
    def from_features(cls, image: Image, features: Features) -> Iterator['QRCode']:
        all_finder_patterns = copy.deepcopy(features.finder_patterns)
        all_alignment_patterns = copy.deepcopy(features.alignment_patterns)

        for finders in group_finder_patterns(all_finder_patterns):
            assert len(finders) == 3

            ordered_finders = orientate_finder_patterns(finders)
            version = guess_version_from_finders(ordered_finders)
            # plt.figure()
            # plt.imshow(image)
            alignment_patterns = choose_and_order_alignments(
                ordered_finders,
                version,
                all_alignment_patterns
            )
            # plt.show()
            # TODO Find fourth corner

            yield QRCode(
                image=image,
                finder_patterns=ordered_finders,
                version=version,
                alignment_patterns=alignment_patterns,
                fourth_corner=None
            )

    def get_bounding_box_image(self):
        f1, f2, f3 = self.finder_patterns
        v1 = f2.corners[1] - f1.corners[0]
        v2 = f3.corners[3] - f1.corners[0]
        det = v1[0] * v2[1] - v1[1] * v2[0]
        mat = np.array([
            [v2[1] / det, - v2[0] / det, 0],
            [- v1[1] / det, v1[0] / det, 0],
            [0, 0, 1]
        ]) @ np.array([
            [1, 0, - f1.center[0]],
            [0, 1, - f1.center[1]],
            [0, 0, 1]
        ])
        mat_inv = np.linalg.inv(mat)

        corners_relative = [
            [-0.25, -0.25],
            [-0.25, 1.25],
            [1.25, -0.25],
            [1.25, 1.25]
        ]
        corners = np.array([
            (mat_inv @ np.array([
                [c[0]],
                [c[1]],
                [1]
            ])).T[0].tolist()
            for c in corners_relative
        ], dtype=np.int)[:, :-1]

        bbox = create_bounding_box(corners)
        bbox[bbox < 0] = 0
        for i, v in enumerate(bbox[:, 0]):
            s = self.image.shape[0]
            if s < v:
                bbox[i, 0] = s
        for i, v in enumerate(bbox[:, 1]):
            s = self.image.shape[1]
            if s < v:
                bbox[i, 1] = s

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]])
        # plt.show()

        return self.image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]

    def create_references(self, features: List[MatchingFeatures]) -> References:
        return References(
            features=features,
            alignments_found=[a is not None for a in self.alignment_patterns],
            fourth_corner_found=self.fourth_corner is not None
        )

    def get_references(self, references: References) -> np.ndarray:
        match_features = self._feature_to_points(references)

        return np.concatenate([
            match_features[feature][1]
            for feature in references.features
            if match_features[feature][0]
        ], axis=0)

    def _feature_to_points(self, references: References) -> Dict:
        return {
            MatchingFeatures.FINDER_CENTERS: (True, np.array([
                f.center
                for f in self.finder_patterns
            ])),
            MatchingFeatures.FINDER_CORNERS: (True, np.array([
                c
                for f in self.finder_patterns
                for c in f.corners
            ])),
            MatchingFeatures.ALIGNMENTS_CENTERS: (any(references.alignments_found), np.array([
                a.center
                for a in self.alignment_patterns
                if a is not None
            ])),
            MatchingFeatures.FOURTH_CORNER: (references.fourth_corner_found, np.array([
                self.fourth_corner
            ]))
        }

    def correct(self, method: Optional[Correction] = None, **kwargs):
        from tfginfo.transformations import correction
        return correction(self, method, **kwargs)

    def binarize(self, **kwargs):
        from tfginfo.transformations import binarization
        return binarization(self, **kwargs)

    def decode(self, bounding_box: bool = True):
        if bounding_box:
            image = self.get_bounding_box_image()
        else:
            image = self.image

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image)
        # plt.show()

        results = decode(image)

        assert len(results) > 0
        if len(results) > 1:
            raise ValueError("More than one QR found in the bbox of the QR")

        return results[0]

    def sample(self):
        qr_sampled = copy.deepcopy(self)
        qr_sampled.binarize()
        qr_sampled.correct(method=Correction.PROJECTIVE, bitpixel=1, border=0)

        return SampledQRCode(
            image=qr_sampled.image,
            version=qr_sampled.version
        )

    def plot(self, axes=None, interpolation: Optional[str] = None):
        finders_centers = np.array([f.center for f in self.finder_patterns])
        aligns_centers = np.array([a.center
                                   for a in self.alignment_patterns
                                   if a is not None])
        corners = np.array([finder.corners for finder in self.finder_patterns])

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)

        axes.imshow(self.image, interpolation=interpolation)
        axes.scatter(*finders_centers[:, ::-1].T)
        if len(aligns_centers) > 0:
            axes.scatter(*aligns_centers[:, ::-1].T)

        for corn in corners:
            axes.scatter(corn[0][1], corn[0][0], color="red")
            axes.scatter(corn[1][1], corn[1][0], color="green")
            axes.scatter(corn[2][1], corn[2][0], color="blue")
            axes.scatter(corn[3][1], corn[3][0], color="brown")

        if self.fourth_corner is not None:
            axes.scatter(self.fourth_corner[1], self.fourth_corner[0])

        for num, fc in enumerate(self.finder_patterns):
            c = fc.center
            s = fc.hratios[2] // 3
            axes.text(c[1] - s, c[0] + s, str(num + 1), color="red", fontsize="x-large")

        for num, ap in enumerate(self.alignment_patterns):
            if ap is None:
                continue

            c = ap.center
            s = ap.hratios[1]
            axes.text(c[1] - s, c[0] + s, str(num + 1), color="red", fontsize="large")


@dataclass
class SampledQRCode:
    image: Image
    version: int

    def count_errors(self, original: np.ndarray) -> int:
        mask_2d = (img_as_ubyte(color.rgb2gray(self.image)) !=
                   img_as_ubyte(color.rgb2gray(original)))

        return len(np.nonzero(mask_2d)[0])

    def plot(self, axes=None) -> None:
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)

        axes.imshow(self.image, interpolation=None)

    def plot_differences(self, matrix: np.array, axes=None) -> None:
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)

        mask_2d = (img_as_ubyte(color.rgb2gray(self.image)) !=
                   img_as_ubyte(color.rgb2gray(matrix)))
        masked = np.ma.masked_equal(mask_2d, 0)

        cm = matplotlib.colors.LinearSegmentedColormap.from_list(
            "constant_red", [(1, 0, 0), (1, 0, 0)], N=2
        )
        axes.imshow(self.image, alpha=0.2, interpolation="none")
        axes.imshow(
            masked,
            alpha=0.8,
            cmap=cm,
            interpolation="none",
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
        )
