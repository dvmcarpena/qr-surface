from functools import wraps
from typing import Callable, List, Optional

import numpy as np
from skimage import img_as_ubyte, transform

from tfginfo.features import AlignmentPattern, FinderPattern
from tfginfo.qr import QRCode, IdealQRCode
from tfginfo.matching import MatchingFeatures
from tfginfo.utils import rgb2binary

Transformation = Callable[..., QRCode]


def apply_linear_transformation(qr: QRCode, linear_map, ideal_qr: IdealQRCode) -> QRCode:
    qr.image = img_as_ubyte(transform.warp(
        image=qr.image,
        inverse_map=linear_map.inverse,
        output_shape=(ideal_qr.size, ideal_qr.size),
        order=0,
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
        # qr.fourth_corner = linear_map(qr.fourth_corner[::-1])[0][::-1]

    return qr


def apply_nonlinear_transformation(qr: QRCode, inverse_map, ideal_qr: IdealQRCode) -> QRCode:
    qr.image = img_as_ubyte(transform.warp(
        image=qr.image,
        inverse_map=inverse_map,
        output_shape=(ideal_qr.size, ideal_qr.size),
        order=3,
        mode="edge"
    ))

    bw_image = rgb2binary(qr.image)

    qr.finder_patterns = tuple([
        FinderPattern.from_center_and_ratios(
            image=bw_image,
            center=dst_center[::-1],
            hratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel,
            vratios=np.array([1, 1, 3, 1, 1]) * ideal_qr.bitpixel
        )
        for f, dst_center in zip(qr.finder_patterns, ideal_qr.finders_centers)
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

    return qr


def general_correction(is_lineal: bool, build_transformation_function: Callable,
                       default_references_features: List[MatchingFeatures]) -> Callable:

    @wraps(build_transformation_function)
    def instance_correction(qr: QRCode, bitpixel: int, border: int,
                            references_features: Optional[List[MatchingFeatures]] = None) -> QRCode:
        if bitpixel % 2 == 0:
            raise ValueError("Bitpixel needs to be a odd number")
        if references_features is None:
            references_features = default_references_features

        references = qr.create_references(references_features)
        ideal_qr = IdealQRCode(qr.version, bitpixel, border)

        src = qr.get_references(references)
        dst = ideal_qr.get_references(references)
        # print(qr.version)
        # print(dst)

        if is_lineal:
            linear_map = build_transformation_function(src, dst)
            return apply_linear_transformation(
                qr=qr,
                linear_map=linear_map,
                ideal_qr=ideal_qr
            )
        else:
            inverse_map = build_transformation_function(qr, src, dst, ideal_qr.size)
            return apply_nonlinear_transformation(
                qr=qr,
                inverse_map=inverse_map,
                ideal_qr=ideal_qr
            )

    return instance_correction
