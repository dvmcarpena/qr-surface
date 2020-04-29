from typing import Callable, Optional

import numpy as np
from skimage import transform

from tfginfo.qr import QRCode, Correction
from tfginfo.utils import rgb2binary2rgb
from tfginfo.matching import MatchingFeatures

from .interpolate import RadialBasisSplines
from .general import general_correction

Transformation = Callable[..., QRCode]


def binarization(qr: QRCode, block_size: Optional[int] = None, **kwargs) -> QRCode:
    qr.image = rgb2binary2rgb(qr.image, block_size=block_size, **kwargs)
    return qr


def affine_correction(src: np.ndarray, dst: np.ndarray):
    return transform.estimate_transform(
        ttype="affine",
        src=src[:, ::-1],
        dst=dst
    )


def projective_correction(src: np.ndarray, dst: np.ndarray):
    return transform.estimate_transform(
        ttype="projective",
        src=src[:, ::-1],
        dst=dst
    )


def cylindrical_transformation(qr: QRCode, src: np.ndarray, dst: np.ndarray, dst_size: int):
    pass


def tps_transformation(qr: QRCode, src: np.ndarray, dst: np.ndarray, dst_size: int):
    tps_inv_tps_m = RadialBasisSplines(
        source_landmarks=dst[:, ::-1],
        destiny_landmarks=src[:, ::-1],
        function="thin_plate",
        norm="euclidean",
        # smooth=1
    )

    markers = np.stack(np.meshgrid(range(dst_size), range(dst_size)), axis=2)
    markers_inv_tps = tps_inv_tps_m(*markers.reshape((-1, 2)).T)
    return lambda _: markers_inv_tps


_CORRECTION_METHOD_FUNC = {
    Correction.AFFINE: general_correction(
        is_lineal=True,
        build_transformation_function=affine_correction,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.ALIGNMENTS_CENTERS,
            MatchingFeatures.FOURTH_CORNER
        ]
    ),
    Correction.PROJECTIVE: general_correction(
        is_lineal=True,
        build_transformation_function=projective_correction,
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
