from functools import wraps
from typing import Callable, List, Optional, Tuple

import numpy as np
from skimage import img_as_ubyte, color, filters, transform

from tfginfo.features import AlignmentPattern, FinderPattern
from tfginfo.qr import QRCode, Correction
from tfginfo.utils import get_size_from_version, get_alignments_centers
from tfginfo.matching import MatchingFeatures, References

from .interpolate import RadialBasisSplines
from .ideal import IdealQRCode

Transformation = Callable[..., QRCode]


def rgb2binary(image: np.ndarray, block_size: Optional[int] = None) -> np.ndarray:
    block_size = 151 if block_size is None else block_size
    assert isinstance(block_size, int)

    gray_image = color.rgb2gray(image)
    threshold: np.ndarray = filters.threshold_sauvola(gray_image, block_size)
    #threshold = filters.threshold_otsu(gray_image)

    return gray_image > threshold


def rgb2binary2rgb(image: np.ndarray, **kwargs) -> np.ndarray:
    return img_as_ubyte(color.gray2rgb(rgb2binary(image, **kwargs)))


def binarization(qr: QRCode, block_size: Optional[int] = None, **kwargs) -> QRCode:
    qr.image = rgb2binary2rgb(qr.image, block_size=block_size)
    return qr


# def sampling(qr: QRCode, border: int = 12, **kwargs) -> QRCode:
#     pass


# def affine_correction(qr: QRCode, bitpixel: int, border: int) -> QRCode:
#     centers = np.array([f.center for f in qr.finder_patterns])[:, ::-1]
#     size = get_size_from_version(qr.version)
#     bitpixel = 1
#     border = 25
#     corrected_centers = (np.array([
#         [3, 3],
#         [size - 1 - 3, 3],
#         [3, size - 1 - 3]
#     ]) + np.array([border, border])) * bitpixel
#     trans = transform.estimate_transform(
#         ttype="affine",
#         src=centers,
#         dst=corrected_centers
#     )
#
#     output_size = (size + 2 * border) * bitpixel
#
#     # import matplotlib.pyplot as plt
#     # fig, (ax1, ax2) = plt.subplots(1, 2)
#     # ax1.imshow(qr.image)
#     # ax2.imshow(transform.warp(
#     #     image=qr.image,
#     #     inverse_map=trans.inverse,
#     #     output_shape=(output_size, output_size)
#     # ))
#     # plt.show()
#     # qr.image = transform.warp(
#     #     image=qr.image,
#     #     inverse_map=trans.inverse,
#     #     output_shape=(output_size, output_size)
#     # )
#
#     qr.image = img_as_ubyte(transform.warp(
#         image=qr.image,
#         inverse_map=trans.inverse,
#         output_shape=(output_size, output_size),
#         mode="edge"
#     ))
#     qr.finder_patterns = tuple([
#         FinderPattern(
#             center=trans(f.center[::-1])[0][::-1],
#             corners=trans(f.corners[:, ::-1])[:, ::-1],
#             contour=trans(f.contour[:, ::-1])[:, ::-1],
#             hratios=np.array([1, 1, 3, 1, 1]) * bitpixel,
#             vratios=np.array([1, 1, 3, 1, 1]) * bitpixel
#         )
#         for f in qr.finder_patterns
#     ])
#     qr.alignment_patterns = [
#         AlignmentPattern(
#             center=trans(a.center[::-1])[0][::-1],
#             corners=trans(a.corners[:, ::-1])[:, ::-1],
#             contour=trans(a.contour[:, ::-1])[:, ::-1],
#             hratios=np.array([1, 1, 3, 1, 1]) * bitpixel,
#             vratios=np.array([1, 1, 3, 1, 1]) * bitpixel
#         ) if a is not None else None
#         for a in qr.alignment_patterns
#     ]
#
#     return qr


# def projective_correction(qr: QRCode, bitpixel: int, border: int) -> QRCode:
#     finders_centers = [f.center for f in qr.finder_patterns]
#     finder_corners = [c for f in qr.finder_patterns for c in f.corners]
#     alignments_centers = [
#         a.center
#         for a in qr.alignment_patterns
#         if a is not None
#     ]
#     alignments_found = np.array([a is not None for a in qr.alignment_patterns])
#     src = np.array(finders_centers + alignments_centers)[:, ::-1]
#
#     size = get_size_from_version(qr.version)
#     corrected_centers = (np.array([
#         [3, 3],
#         [size - 1 - 3, 3],
#         [3, size - 1 - 3]
#     ]) + np.array([border, border])) * bitpixel
#     corners = (np.array([
#         [0, 0],
#         [7, 0],
#         [7, 7],
#         [0, 7],
#         [size - 7, 0],
#         [size, 0],
#         [size, 7],
#         [size - 7, 7],
#         [0, size - 7],
#         [7, size - 7],
#         [7, size],
#         [0, size],
#     ]) + np.array([border, border])) * bitpixel
#     ideal_positions = get_alignments_centers(qr.version)
#     ideal_positions = np.array(ideal_positions)
#     ideal_positions = ideal_positions[np.lexsort((ideal_positions[:, 0],
#                                                   ideal_positions[:, 1]))]
#     ideal_positions = ideal_positions[alignments_found]
#     ideal_positions = (ideal_positions + np.array([border, border])) * bitpixel
#     dst = np.concatenate((corrected_centers, ideal_positions), axis=0)
#
#     # Alignments
#     trans = transform.estimate_transform(
#         ttype="projective",
#         src=src,
#         dst=dst
#     )
#
#     output_size = (size + 2 * border) * bitpixel
#
#     # import matplotlib.pyplot as plt
#     # fig, (ax1, ax2) = plt.subplots(1, 2)
#     # ax1.imshow(qr.image)
#     # ax2.imshow(transform.warp(
#     #     image=qr.image,
#     #     inverse_map=trans.inverse,
#     #     output_shape=(output_size, output_size)
#     # ))
#     # plt.show()
#
#     qr.image = img_as_ubyte(transform.warp(
#         image=qr.image,
#         inverse_map=trans.inverse,
#         output_shape=(output_size, output_size),
#         order=0,
#         mode="edge"
#     ))
#     qr.finder_patterns = tuple([
#         FinderPattern(
#             center=trans(f.center[::-1])[0][::-1],
#             corners=trans(f.corners[:, ::-1])[:, ::-1],
#             contour=trans(f.contour[:, ::-1])[:, ::-1],
#             hratios=np.array([1, 1, 3, 1, 1]) * bitpixel,
#             vratios=np.array([1, 1, 3, 1, 1]) * bitpixel
#         )
#         for f in qr.finder_patterns
#     ])
#     qr.alignment_patterns = [
#         AlignmentPattern(
#             center=trans(a.center[::-1])[0][::-1],
#             corners=(trans(a.corners[:, ::-1])[:, ::-1] if a.corners is not None else None),
#             contour=(trans(a.contour[:, ::-1])[:, ::-1] if a.contour is not None else None),
#             hratios=np.array([1, 1, 1]) * bitpixel,
#             vratios=np.array([1, 1, 1]) * bitpixel
#         ) if a is not None else None
#         for a in qr.alignment_patterns
#     ]
#
#     return qr


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


# def cylindrical_correction(qr: QRCode, bitpixel: int, border: int,
#                            references_features: Optional[List[MatchingFeatures]] = None) -> QRCode:
#     references_features = references_features if references_features is not None else [
#         MatchingFeatures.FINDER_CENTERS,
#         MatchingFeatures.FINDER_CORNERS,
#         MatchingFeatures.FOURTH_CORNER,
#         MatchingFeatures.ALIGNMENTS_CENTERS,
#     ]
#
#     references = qr.create_references(references_features)
#     ideal_qr = IdealQRCode(qr.version, bitpixel, border)
#
#     # TODO
#     pass
#
#     # return apply_nonlinear_transformation(
#     #     qr=qr,
#     #     inverse_map=lambda _: markers_inv_tps,
#     #     ideal_qr=ideal_qr,
#     #     bitpixel=bitpixel
#     # )


def affine_correction(src: np.ndarray, dst: np.ndarray):
    return transform.estimate_transform(
        ttype="affine",
        src=src[:, ::-1],
        dst=dst
    )


def new_proj(src: np.ndarray, dst: np.ndarray):
    return transform.estimate_transform(
        ttype="projective",
        src=src[:, ::-1],
        dst=dst
    )


def cylindrical_transformation(src: np.ndarray, dst: np.ndarray, dst_size: int):
    pass


def tps_transformation(src: np.ndarray, dst: np.ndarray, dst_size: int):
    tps_inv_tps_m = RadialBasisSplines(
        source_landmarks=dst[:, ::-1],
        destiny_landmarks=src[:, ::-1],
        function="thin_plate",
        norm="euclidean",
        #smooth=1
    )

    markers = np.stack(np.meshgrid(range(dst_size), range(dst_size)), axis=2)
    markers_inv_tps = tps_inv_tps_m(*markers.reshape((-1, 2)).T)
    return lambda _: markers_inv_tps


def general_correction(is_lineal: bool, build_transformation_function: Callable,
                       default_references_features: List[MatchingFeatures]) -> Callable:

    def instance_correction(qr: QRCode, bitpixel: int, border: int,
                   references_features: Optional[List[MatchingFeatures]] = None) -> QRCode:
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
            inverse_map = build_transformation_function(src, dst, ideal_qr.size)
            return apply_nonlinear_transformation(
                qr=qr,
                inverse_map=inverse_map,
                ideal_qr=ideal_qr
            )

    return instance_correction


# def tps_correction(qr: QRCode, bitpixel: int, border: int,
#                    references_features: Optional[List[MatchingFeatures]] = None) -> QRCode:
#     references_features = references_features if references_features is not None else [
#         MatchingFeatures.FINDER_CENTERS,
#         MatchingFeatures.FINDER_CORNERS,
#         MatchingFeatures.FOURTH_CORNER,
#         MatchingFeatures.ALIGNMENTS_CENTERS,
#     ]
#
#     references = qr.create_references(references_features)
#     ideal_qr = IdealQRCode(qr.version, bitpixel, border)
#
#     src = qr.get_references(references)
#     dst = ideal_qr.get_references(references)
#
#     inverse_map = tps_transformation(src, dst, ideal_qr.size)
#
#     return apply_nonlinear_transformation(
#         qr=qr,
#         inverse_map=inverse_map,
#         ideal_qr=ideal_qr
#     )


_CORRECTION_METHOD_FUNC = {
    Correction.AFFINE: affine_correction,
    # Correction.PROJECTIVE: projective_correction,
    Correction.PROJECTIVE: general_correction(
        is_lineal=True,
        build_transformation_function=new_proj,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.ALIGNMENTS_CENTERS,
            MatchingFeatures.FOURTH_CORNER
        ]
    ),
    Correction.CYLINDRICAL: general_correction(
        is_lineal=False,
        build_transformation_function=cylindrical_transformation,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.FINDER_CORNERS,
            MatchingFeatures.ALIGNMENTS_CENTERS,
            MatchingFeatures.FOURTH_CORNER
        ]
    ),
    Correction.TPS: general_correction(
        is_lineal=False,
        build_transformation_function=tps_transformation,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.FINDER_CORNERS,
            MatchingFeatures.ALIGNMENTS_CENTERS,
            MatchingFeatures.FOURTH_CORNER
        ]
    )
}


def correction(qr: QRCode, method: Optional[Correction] = None, bitpixel: int = 11, border: int = 15,
               **kwargs) -> QRCode:
    if method is None:
        method = Correction.PROJECTIVE

    func = _CORRECTION_METHOD_FUNC[method]
    return func(qr, bitpixel, border, **kwargs)
