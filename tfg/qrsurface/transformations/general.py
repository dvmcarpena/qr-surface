from functools import wraps
from typing import Callable, List, Optional

import numpy as np
from skimage import img_as_ubyte, transform

from ..features import AlignmentPattern, FinderPattern
from ..qr import QRCode, IdealQRCode
from ..matching import MatchingFeatures
from ..utils import rgb2binary

Transformation = Callable[..., QRCode]


def apply_linear_transformation(qr: QRCode, linear_map, ideal_qr: IdealQRCode) -> QRCode:
    """
    Application of a linear transformation to the QR Code

    :param qr: The QRCode object
    :param linear_map: The linear forward mapping of the transformation
    :param ideal_qr: The ideal QR Code correspondring to the destiny

    :return: The corrected QRCode object
    """
    qr.image = img_as_ubyte(transform.warp(
        image=qr.image,
        inverse_map=linear_map.inverse,
        output_shape=(ideal_qr.size, ideal_qr.size),
        order=3,
        mode="edge"
    ))

    qr.finder_patterns = tuple([
        FinderPattern(
            center=linear_map(f.center[::-1])[0][::-1],
            corners=linear_map(f.corners[:, ::-1])[:, ::-1],
            contour=linear_map(f.contour[:, ::-1])[:, ::-1],
            hratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel,
            vratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel
        )
        for f in qr.finder_patterns
    ])
    qr.alignment_patterns = [
        AlignmentPattern(
            center=linear_map(a.center[::-1])[0][::-1],
            corners=(linear_map(a.corners[:, ::-1])[:, ::-1] if a.corners is not None else None),
            contour=(linear_map(a.contour[:, ::-1])[:, ::-1] if a.contour is not None else None),
            hratios=np.array([1, 1, 1]) * ideal_qr.bitpixel,
            vratios=np.array([1, 1, 1]) * ideal_qr.bitpixel
        ) if a is not None else None
        for a in qr.alignment_patterns
    ]
    if qr.fourth_corner is not None:
        qr.fourth_corner = ideal_qr.fourth_corner

    return qr


def apply_nonlinear_transformation(qr: QRCode, inverse_map, ideal_qr: IdealQRCode, simple: bool) -> QRCode:
    """
    Application of a non-linear transformation to the QR Code

    :param qr: The QRCode object
    :param inverse_map: The inverse mapping of the transformation
    :param ideal_qr: The ideal QR Code corresponding to the destiny
    :param simple: Whether we want to scan the image for the destiny features or assume that they have landed in the
        designated spots of the destiny references

    :return: The corrected QRCode object
    """
    qr.image = img_as_ubyte(transform.warp(
        image=qr.image,
        inverse_map=inverse_map,
        output_shape=(ideal_qr.size, ideal_qr.size),
        order=3,
        mode="edge"
    ))

    if simple:
        qr.finder_patterns = tuple([
            FinderPattern(
                center=center,
                corners=corners,
                contour=corners,
                hratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel,
                vratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel
            )
            for center, corners in zip(ideal_qr.finders_centers, ideal_qr.finders_corners_by_finder)
        ])
        # noinspection PyTypeChecker
        qr.alignment_patterns = [
            AlignmentPattern(
                center=center,
                corners=None,
                contour=None,
                hratios=np.array([1, 1, 1]) * ideal_qr.bitpixel,
                vratios=np.array([1, 1, 1]) * ideal_qr.bitpixel
            )
            for center in ideal_qr.alignments_centers
        ]
        if qr.fourth_corner is not None:
            qr.fourth_corner = ideal_qr.fourth_corner
    else:
        bw_image = rgb2binary(qr.image)

        try:
            qr.finder_patterns = tuple([
                FinderPattern.from_center_and_ratios(
                    image=bw_image,
                    center=dst_center[::-1],
                    hratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel,
                    vratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel
                )
                for f, dst_center in zip(qr.finder_patterns, ideal_qr.finders_centers)
            ])
        except ValueError:
            qr.finder_patterns = tuple([
                FinderPattern(
                    center=center,
                    corners=corners,
                    contour=corners,
                    hratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel,
                    vratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel
                )
                for center, corners in zip(ideal_qr.finders_centers, ideal_qr.finders_corners_by_finder)
            ])
        qr.alignment_patterns = [
            AlignmentPattern.from_center_and_ratios(
                image=bw_image,
                center=dst_center[::-1],
                hratios=np.array([1, 1, 1]) * ideal_qr.bitpixel,
                vratios=np.array([1, 1, 1]) * ideal_qr.bitpixel
            ) if a is not None else None
            for a, dst_center in zip(qr.alignment_patterns, ideal_qr.alignments_centers)
        ]
        if qr.fourth_corner is not None:
            qr.fourth_corner = ideal_qr.fourth_corner

        if any([len(finder.corners) != 4 for finder in qr.finder_patterns]):
            qr.finder_patterns = tuple([
                FinderPattern(
                    center=center,
                    corners=corners,
                    contour=corners,
                    hratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel,
                    vratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel
                )
                for center, corners in zip(ideal_qr.finders_centers, ideal_qr.finders_corners_by_finder)
            ])

    return qr


def general_correction(is_lineal: bool, build_transformation_function: Callable,
                       default_references_features: List[MatchingFeatures]) -> Callable:
    """
    Abstract method which build a general correction using the build transformation function and if the transformation
    is linear or not.

    :param is_lineal: Whether the transformation is linear
    :param build_transformation_function: A function which if called return a inverse mapping
    :param default_references_features: The default selection of references for this method

    :return: A function which executes the correction of a given QRCode
    """

    @wraps(build_transformation_function)
    def instance_correction(qr: QRCode, bitpixel: int, border: int,
                            references_features: Optional[List[MatchingFeatures]] = None,
                            simple: bool = False) -> QRCode:
        if bitpixel % 2 == 0:
            raise ValueError("Bitpixel needs to be a odd number")
        if references_features is None:
            references_features = default_references_features

        references = qr.create_references(references_features)
        ideal_qr = IdealQRCode(qr.version, bitpixel, border)

        src = qr.get_references(references)
        dst = ideal_qr.get_references(references)

        if is_lineal:
            linear_map = build_transformation_function(src, dst)
            return apply_linear_transformation(
                qr=qr,
                linear_map=linear_map,
                ideal_qr=ideal_qr
            )
        else:
            inverse_map = build_transformation_function(qr, src, dst, ideal_qr)
            return apply_nonlinear_transformation(
                qr=qr,
                inverse_map=inverse_map,
                ideal_qr=ideal_qr,
                simple=simple
            )

    return instance_correction
