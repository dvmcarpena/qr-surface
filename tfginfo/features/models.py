from dataclasses import dataclass

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
        test_point_xoffset = hratios[1] + (hratios[0] + hratios[2]) // 2
        contour = get_contour_from_center_and_ratios(image, center, hratios, vratios,
                                                     test_point_xoffset)
        corners = get_corners_from_contour(
            contour,
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios))) // 10
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
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios))) // 10
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
        test_point_xoffset = (hratios[0] + hratios[1]) // 2
        #try:
        contour = get_contour_from_center_and_ratios(image, center, hratios, vratios,
                                                     test_point_xoffset)
        corners = get_corners_from_contour(
            contour,
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios))) // 10
        )
        #except ValueError as e:
        #    print(e)
        #    contour = None
        #    corners = None
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
            image.shape,
            num_corners=4,
            min_distance=min(map(sum, (hratios, vratios))) // 10
        )
        return AlignmentPattern(
            center=center,
            contour=contour,
            corners=corners,
            hratios=hratios,
            vratios=vratios
        )
