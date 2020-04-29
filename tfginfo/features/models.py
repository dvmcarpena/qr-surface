from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from tfginfo.utils import Array, Image
from .utils import (get_center_from_contour, get_contour_from_center_and_ratios,
                    get_corners_from_contour, get_ratios_from_center)


@dataclass
class FinderPattern:
    center: Array
    contour: Array
    corners: Array
    hratios: Array
    vratios: Array

    @classmethod
    def from_center_and_ratios(cls, image: Image, center: Array, hratios: Array,
                               vratios: Array) -> 'FinderPattern':
        # test_point_xoffset = hratios[1] + (hratios[0] + hratios[2]) // 2
        contour = get_contour_from_center_and_ratios(image, center, hratios, vratios)
        corners = get_corners_from_contour(
            contour,
            center,
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios)))
        )
        return FinderPattern(
            center=center,
            contour=contour,
            corners=corners,
            hratios=hratios,
            vratios=vratios
        )

    @classmethod
    def from_contour(cls, image: Image, contour: Array) -> 'FinderPattern':
        center = get_center_from_contour(contour, image.shape)
        hratios, vratios = get_ratios_from_center(image, center)
        corners = get_corners_from_contour(
            contour,
            center,
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios)))
        )
        return FinderPattern(
            center=center,
            contour=contour,
            corners=corners,
            hratios=hratios,
            vratios=vratios
        )


@dataclass
class AlignmentPattern:
    center: Array
    contour: Array
    corners: Array
    hratios: Array
    vratios: Array

    @classmethod
    def from_center_and_ratios(cls, image: Image, center: Array, hratios: Array,
                               vratios: Array) -> 'AlignmentPattern':
        # try:
        #     # contour = get_contour_from_center_and_ratios(image, center, hratios, vratios)
        #     # corners = get_corners_from_contour(
        #     #     contour,
        #     #     center,
        #     #     image.shape,
        #     #     num_corners=4,
        #     #     min_distance=min(map(sum, (hratios, vratios)))
        #     # )
        # except Exception as e:
        #    print(e)
        #    contour = None
        #    corners = None
        contour = None
        corners = None
        return AlignmentPattern(
            center=center,
            contour=contour,
            corners=corners,
            hratios=hratios,
            vratios=vratios
        )

    @classmethod
    def from_contour(cls, image: Image, contour: Array) -> 'AlignmentPattern':
        center = get_center_from_contour(contour, image.shape)
        hratios, vratios = get_ratios_from_center(image, center)
        corners = get_corners_from_contour(
            contour,
            center,
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios)))
        )
        return AlignmentPattern(
            center=center,
            contour=contour,
            corners=corners,
            hratios=hratios,
            vratios=vratios
        )


class Features:

    def __init__(self, image: Image,
                 finder_patterns: List[FinderPattern],
                 alignment_patterns: List[AlignmentPattern]) -> None:
        self.image = image
        self.finder_patterns = finder_patterns
        self.alignment_patterns = alignment_patterns

    @classmethod
    def from_image(self, image: Image, **kwargs) -> 'Features':
        from .features import find_all_features
        return find_all_features(image, **kwargs)

    def plot(self, axes=None, show=False):
        centroid = np.array([finder.center for finder in self.finder_patterns])
        contours = np.array([finder.contour for finder in self.finder_patterns])
        corns = np.array([finder.corners for finder in self.finder_patterns])

        ap_centers = np.array([align.center for align in self.alignment_patterns])
        ap_contour = np.array([align.contour
                               for align in self.alignment_patterns
                               if align.contour is not None])
        ap_corners = np.array([align.corners
                               for align in self.alignment_patterns
                               if align.corners is not None])

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
        # plt.title(name)
        axes.imshow(self.image)
        if len(centroid) > 0:
            axes.scatter(*centroid[:, ::-1].T)
        if len(ap_centers) > 0:
            axes.scatter(*ap_centers[:, ::-1].T)

        for corn in corns:
            axes.scatter(*corn[:, ::-1].T)
        for corn in ap_corners:
            axes.scatter(*corn[:, ::-1].T)
        for cnt in contours:
            axes.plot(cnt[:, 1], cnt[:, 0], linewidth=2)
        for cnt in ap_contour:
            axes.plot(cnt[:, 1], cnt[:, 0], linewidth=2)

        if show:
            plt.show()
