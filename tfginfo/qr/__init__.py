import copy
from dataclasses import dataclass, field
from enum import auto, Enum, unique
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np
from skimage import color, filters, img_as_ubyte, transform

from tfginfo.utils import Array, Image, create_bounding_box, get_size_from_version, get_alignments_centers
from tfginfo.features import AlignmentPattern, Features, FinderPattern
from tfginfo.decode import decode
from tfginfo.matching import MatchingFeatures, References

from .ideal import IdealQRCode
from .utils import (choose_and_order_alignments, group_finder_patterns, guess_version, guess_version_from_finders,
                    orientate_finder_patterns, OrderedFinderPatterns, find_fourth_corner)


@unique
class Correction(Enum):
    AFFINE = auto()
    PROJECTIVE = auto()
    CYLINDRICAL = auto()
    TPS = auto()
    DOUBLE_TPS = auto()


@dataclass
class QRCode:
    image: Image
    finder_patterns: OrderedFinderPatterns
    version: int
    alignment_patterns: List[Optional[AlignmentPattern]]
    fourth_corner: Optional[Array]
    size: int = field(init=False)

    def __post_init__(self):
        self.size = get_size_from_version(self.version)

    @classmethod
    def from_image(cls, image: Image, **kwargs) -> Iterator['QRCode']:
        features = Features.from_image(image, **kwargs)
        return cls.from_features(image, features)

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

            fourth_corner = find_fourth_corner(
                features.bw_image,
                ordered_finders,
                version,
                all_alignment_patterns
            )

            yield QRCode(
                image=image,
                finder_patterns=ordered_finders,
                version=version,
                alignment_patterns=alignment_patterns,
                fourth_corner=fourth_corner
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

    def decode(self, bounding_box: bool = True, sample: bool = False):
        if bounding_box:
            image = self.get_bounding_box_image()
        elif sample:
            qr_sampled = copy.deepcopy(self)
            qr_sampled.binarize()
            qr_sampled.correct(method=Correction.PROJECTIVE, bitpixel=5, border=5)
            # qr_sampled.correct(method=Correction.PROJECTIVE, bitpixel=5, border=5, references_features=[
            #     MatchingFeatures.FINDER_CENTERS,
            #     MatchingFeatures.FINDER_CORNERS,
            #     MatchingFeatures.ALIGNMENTS_CENTERS,
            #     MatchingFeatures.FOURTH_CORNER
            # ])
            image = qr_sampled.image
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

    def sample(self, method=Correction.PROJECTIVE):
        qr_sampled = copy.deepcopy(self)

        # if len(list(filter(lambda a: a is not None, qr_sampled.alignment_patterns))) == 0:
        #     qr_sampled.correct(method=Correction.PROJECTIVE, bitpixel=1, border=0, references_features=[
        #         MatchingFeatures.FINDER_CENTERS,
        #         MatchingFeatures.FINDER_CORNERS,
        #         MatchingFeatures.ALIGNMENTS_CENTERS,
        #         MatchingFeatures.FOURTH_CORNER
        #     ])
        # if method == Correction.TPS:
        #     qr_sampled.correct(method=method, bitpixel=5, border=0, simple=True)
        #     qr_sampled.binarize()
        #     size = get_size_from_version(qr_sampled.version)
        #     qr_sampled.image = img_as_ubyte(transform.resize(qr_sampled.image, (size, size), order=0, anti_aliasing=False))
        #
        #     # gray_image = color.rgb2gray(qr_sampled.image)
        #     # threshold = filters.threshold_otsu(gray_image)
        #     # qr_sampled.image = img_as_ubyte(color.gray2rgb(gray_image > threshold))
        #     # plt.figure()
        #     # plt.imshow(qr_sampled.image)
        #     # plt.show()
        # else:
        #     qr_sampled.binarize()
        #     qr_sampled.correct(method=method, bitpixel=1, border=0)
        # qr_sampled.correct(method=method, bitpixel=5, border=10)
        # qr_sampled.binarize()
        # qr_sampled.correct(method=method, bitpixel=1, border=0, simple=True)

        qr_sampled.correct(method=method, bitpixel=5, border=0, simple=True)
        qr_sampled.binarize()
        qr_sampled.image = img_as_ubyte(transform.resize(
            image=qr_sampled.image,
            output_shape=(qr_sampled.size, qr_sampled.size),
            order=0,
            anti_aliasing=False
        ))

        return SampledQRCode(
            image=qr_sampled.image,
            version=qr_sampled.version
        )

    def plot(self, axes=None, interpolation: Optional[str] = None, show: bool = False):
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

        if show:
            plt.show()


@dataclass
class SampledQRCode:
    image: Image
    version: int

    def count_errors(self, original: np.ndarray) -> int:
        mask_2d = (img_as_ubyte(color.rgb2gray(self.image)) !=
                   img_as_ubyte(color.rgb2gray(original)))

        return len(np.nonzero(mask_2d)[0])

    def plot(self, axes=None, show: bool = False) -> None:
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)

        axes.imshow(self.image, interpolation=None)

        if show:
            plt.show()

    def plot_differences(self, matrix: np.array, axes=None, show: bool = False) -> None:
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

        if show:
            plt.show()

    def decode(self):
        image = color.rgb2gray(self.image)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image)

        # size = image.shape[0]
        # new_image2 = [None] * size
        #
        # for row in range(size):
        #     new_image2[row] = [None] * size
        #     for col in range(size):
        #         new_image2[row][col] = False

        self._paint_timing_pattern(image, image.shape[0])
        self._paint_alignment_pattern(image, image.shape[0])
        self._paint_postion_pattern(image, image.shape[0])
        # plt.figure()
        # plt.imshow(image)
        # plt.show()

        frame_width = 4
        new_size = (image.shape[0] + 2 * frame_width, image.shape[1] + 2 * frame_width, 3)
        new_image = np.ones(new_size, np.uint8) * 255

        new_image[frame_width:image.shape[0] + frame_width,
                  frame_width:image.shape[1] + frame_width, :] = img_as_ubyte(color.gray2rgb(image))
        new_image = img_as_ubyte(transform.rescale(
            image=new_image,
            scale=5,
            order=0,
            anti_aliasing=False,
            multichannel=True
        ))

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(new_image)
        # plt.show()

        results = decode(new_image)

        assert len(results) > 0
        if len(results) > 1:
            raise ValueError("More than one QR found in the bbox of the QR")

        return results[0]

    def _paint_postion_pattern(self, image, size):
        for row, col in [(0, 0), (size - 7, 0), (0, size - 7)]:
            for r in range(-1, 8):
                if row + r <= -1 or size <= row + r:
                    continue

                for c in range(-1, 8):

                    if col + c <= -1 or size <= col + c:
                        continue

                    if (0 <= r and r <= 6 and (c == 0 or c == 6)
                            or (0 <= c and c <= 6 and (r == 0 or r == 6))
                            or (2 <= r and r <= 4 and 2 <= c and c <= 4)):
                        image[row + r][col + c] = 0
                    else:
                        image[row + r][col + c] = 1

    def _paint_alignment_pattern(self, image, size):
        pos = get_alignments_centers(self.version)

        for i in range(len(pos)):
            row, col = pos[i]

            for r in range(-2, 3):
                for c in range(-2, 3):
                    if (r == -2 or r == 2 or c == -2 or c == 2 or
                            (r == 0 and c == 0)):
                        image[row + r][col + c] = 0
                    else:
                        image[row + r][col + c] = 1

    def _paint_timing_pattern(self, image, size):
        for r in range(8, size - 8):
            image[r][6] = r % 2
            image[6][r] = r % 2
