from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_contour_from_center_and_ratios, get_corners_from_contour

Array = np.ndarray


@dataclass
class FinderPattern:
    """
    Representation of all the target information about a finder pattern
    """
    center: Array
    contour: Array
    corners: Array
    hratios: Array
    vratios: Array

    @classmethod
    def from_center_and_ratios(cls, image: np.ndarray, center: Array, hratios: Array,
                               vratios: Array) -> 'FinderPattern':
        """
        Given a center and ratios constructs a FinderPattern object

        :param image: Image where we found the alignment
        :param center: The center found
        :param hratios: Horizontal ratios found
        :param vratios: Vertical ratios found

        :return: The FinderPattern object
        """
        contour = get_contour_from_center_and_ratios(image, center, hratios, vratios)
        corners = get_corners_from_contour(
            contour,
            center,
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
    """
    Representation of all the target information about an alignment pattern
    """
    center: Array
    contour: Array
    corners: Array
    hratios: Array
    vratios: Array

    @classmethod
    def from_center_and_ratios(cls, image: np.ndarray, center: Array, hratios: Array,
                               vratios: Array) -> 'AlignmentPattern':
        """
        Given a center and ratios constructs a AlignmentPattern object

        :param image: Image where we found the alignment
        :param center: The center found
        :param hratios: Horizontal ratios found
        :param vratios: Vertical ratios found

        :return: The AlignmentPattern object
        """
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
        return AlignmentPattern(
            center=center,
            contour=None,
            corners=None,
            hratios=hratios,
            vratios=vratios
        )


class Features:
    """
    Object representing an image and all the target features found in it
    """

    def __init__(self, image: np.ndarray, bw_image: np.ndarray, finder_patterns: List[FinderPattern],
                 alignment_patterns: List[AlignmentPattern]) -> None:
        self.image = image
        self.bw_image = bw_image
        self.finder_patterns = finder_patterns
        self.alignment_patterns = alignment_patterns

    @classmethod
    def from_image(cls, image: np.ndarray, **kwargs) -> 'Features':
        """
        Construct a Features object searching for features in a given image

        :param image: Image to search for features
        :param kwargs: Keyword arguments to find_all_features

        :return: A Features object
        """
        from .features import find_all_features
        return find_all_features(image, **kwargs)

    def plot(self, axes=None, show: bool = False):
        """
        Plots the current Features object

        :param axes: Whether to use a given matplotlib axes or create a new one
        :param show: Whether to call to the show method of matplotlib
        """
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
