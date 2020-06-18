from typing import Callable, List, Optional

import numpy as np
from skimage import measure, transform

from ..qr import QRCode, Correction, IdealQRCode
from ..utils import rgb2binary2rgb
from ..matching import MatchingFeatures

from .interpolate import ThinPlateSpline
from .general import general_correction
from .geometry import collinear, project_to_cylinder

Transformation = Callable[..., QRCode]


def binarization(qr: QRCode, block_size: Optional[int] = None, **kwargs) -> QRCode:
    """
    Makes a binarization of the underlying image in the QRCode

    :param qr: The QRCode that we want to binarize
    :param block_size: The block size used in the local threshold
    :param kwargs: Keyword arguments to the rgb2binary2rgb function

    :return: The QR with the image binarized
    """
    qr.image = rgb2binary2rgb(qr.image, block_size=block_size, **kwargs)
    return qr


def affine_correction(src: np.ndarray, dst: np.ndarray):
    """
    Build the affine forward linear mapping of the correction between the given references

    :param src: The source reference points
    :param dst: The destiny reference points

    :return: The forward linear mapping transformation
    """
    return transform.estimate_transform(
        ttype="affine",
        src=src[:, ::-1],
        dst=dst
    )


def projective_correction(src: np.ndarray, dst: np.ndarray):
    """
    Build the projective forward linear mapping of the correction between the given references

    :param src: The source reference points
    :param dst: The destiny reference points

    :return: The forward linear mapping transformation
    """
    return transform.estimate_transform(
        ttype="projective",
        src=src[:, ::-1],
        dst=dst
    )


def cylindrical_transformation(qr: QRCode, src: np.ndarray, dst: np.ndarray, ideal_qr: IdealQRCode):
    """
    Build the cylindrical inverse mapping of the correction between the given references

    :param qr: The QRCode object
    :param src: The source reference points
    :param dst: The destiny reference points
    :param ideal_qr: The ideal QR object that we want to achieve at the destiny

    :return: The inverse mapping transformation
    """
    dst_size = ideal_qr.size
    f1 = qr.finder_patterns[0]
    f2 = qr.finder_patterns[1]
    f3 = qr.finder_patterns[2]

    max_side = max(f1.corners[0][0] - f2.corners[1][0], f1.corners[0][1] - f3.corners[3][1])
    sign = np.sign(
        (f1.corners[0][1] + f2.corners[1][1])
        - (f1.corners[1][1] + f2.corners[0][1])
    )
    ellipse_radius = sign * int(2 * max_side / 3)

    sign2 = abs(f1.corners[0][0] - f2.corners[1][0]) > abs(f1.corners[0][1] - f2.corners[1][1])

    colinear = collinear(f1.corners[0][::-1], f1.corners[1][::-1], f2.corners[1][::-1])
    if not colinear:
        ellipse_points = np.concatenate((
            f1.corners[0][::-1].reshape((1, 2)),
            f1.corners[1][::-1].reshape((1, 2)),
            f2.corners[0][::-1].reshape((1, 2)),
            f2.corners[1][::-1].reshape((1, 2))
        ), axis=0)

        ellipse_points_2 = ellipse_points.copy()
        if sign2:
            ellipse_points_2[:, 0] = 2 * ellipse_radius - ellipse_points_2[:, 0]
        else:
            ellipse_points_2[:, 1] = 2 * ellipse_radius - ellipse_points_2[:, 1]
        ellipse_points = np.concatenate((
            ellipse_points,
            ellipse_points_2
        ))

        ellipse = measure.EllipseModel()
        ellipse.estimate(ellipse_points)
        xc, yc, ellipse_width, height, phi = ellipse.params
        ellipse_center = (xc, yc)

        if sign2:
            cil_side = abs(f1.corners[0][0] - f2.corners[1][0])
            cil_center = ellipse_center[1]
            cil_orig = f1.corners[0][0]
            cil_rad = height / cil_side
        else:
            cil_side = abs(f1.corners[0][1] - f2.corners[1][1])
            cil_center = ellipse_center[0]
            cil_orig = f1.corners[0][1]
            cil_rad = ellipse_width / cil_side

        cil_rel = (cil_center - cil_orig) / cil_side

        dst_side = abs(ideal_qr.finders_corners_by_finder[0][0][0] - ideal_qr.finders_corners_by_finder[1][1][0])
        dst_orig = ideal_qr.finders_corners_by_finder[0][0][0]
        cil_cen = dst_side * cil_rel + dst_orig
        cil_rad *= dst_side
        cylinder_dst = project_to_cylinder(dst, cil_cen, cil_rad)
    else:
        cylinder_dst = np.vstack((dst[:, 0], dst[:, 1], np.full_like(dst[:, 0], 0))).T

    b_u = []
    b_v = []
    d_u = []
    d_v = []
    for (x, y, z), (u, v) in zip(cylinder_dst, src[:, ::-1]):
        row_u = np.array([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z])
        row_v = np.array([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z])
        d_u.append(row_u)
        d_v.append(row_v)
        b_u.append(u)
        b_v.append(v)

    D = np.array(d_u + d_v)
    b = np.array(b_u + b_v)

    # Solve using ls
    m, res, rank, sv = np.linalg.lstsq(D, b, rcond=None)
    matrix = np.append(m, 1).reshape(3, 4)

    xy = np.array(np.meshgrid(np.arange(dst_size), np.arange(dst_size))).T.reshape(-1, 2)
    if not colinear:
        cylinder_dst = project_to_cylinder(xy, cil_cen, cil_rad)
    else:
        cylinder_dst = np.vstack((xy[:, 0], xy[:, 1], np.full_like(xy[:, 0], ellipse_radius))).T
    cylinder_dst = np.hstack((cylinder_dst, np.ones([cylinder_dst.shape[0], 1])))
    UV = matrix @ cylinder_dst.T
    UV /= UV[2]
    return lambda _: UV[:2].T


def tps_transformation(_: QRCode, src: np.ndarray, dst: np.ndarray, ideal_qr: IdealQRCode):
    """
    Build the TPS inverse mapping of the correction between the given references

    :param _: The QRCode object, not needed by this method
    :param src: The source reference points
    :param dst: The destiny reference points
    :param ideal_qr: The ideal QR object that we want to achieve at the destiny

    :return: The inverse mapping transformation
    """
    dst_size = ideal_qr.size
    tps_inv_tps_m = ThinPlateSpline(
        source_landmarks=dst[:, ::-1],
        destiny_landmarks=src[:, ::-1]
    )

    markers = np.stack(np.meshgrid(np.arange(dst_size), np.arange(dst_size)), axis=2)
    markers_inv_tps = tps_inv_tps_m(markers.reshape((-1, 2)))
    return lambda _: markers_inv_tps


def double_tps(qr: QRCode, bitpixel: int, border: int,
               _: Optional[List[MatchingFeatures]] = None,
               simple: bool = False):
    """
    Transformation which makes a doble TPS transformations, with a relocaliztion in the middle

    :param qr: A QRCode object
    :param bitpixel: The number of pixels per module that we want in the output of the correction
    :param border: The number of modules of border that we want in the correction
    :param _: selected matching features, which get ignored by this method
    :param simple: Whether to use simple or complex relocalization of features

    :return: The corrected QRCode
    """
    first_tps_features = [
        MatchingFeatures.FINDER_CENTERS,
        MatchingFeatures.FINDER_CORNERS,
        MatchingFeatures.ALIGNMENTS_CENTERS
    ]
    second_tps_features = [
        MatchingFeatures.FINDER_CENTERS,
        MatchingFeatures.FINDER_CORNERS,
        MatchingFeatures.ALIGNMENTS_CENTERS,
        MatchingFeatures.FOURTH_CORNER
    ]
    qr.correct(method=Correction.TPS, bitpixel=5, border=4, references_features=first_tps_features, simple=True)
    try:
        qrs = list(QRCode.from_image(qr.image))
    except (ValueError, AttributeError):
        qrs = []
    if len(qrs) > 0:
        assert len(qrs) == 1
        new_qr = qrs[0]

        new_qr.correct(method=Correction.TPS, bitpixel=bitpixel, border=border,
                       references_features=second_tps_features, simple=simple)

        qr.image = new_qr.image
        qr.finder_patterns = new_qr.finder_patterns
        qr.alignment_patterns = new_qr.alignment_patterns
        qr.fourth_corner = new_qr.fourth_corner

    return qr


# The dictionary which ties each identifier to their respective functions
_CORRECTION_METHOD_FUNC = {
    Correction.AFFINE: general_correction(
        is_lineal=True,
        build_transformation_function=affine_correction,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.ALIGNMENTS_CENTERS
        ]
    ),
    Correction.PROJECTIVE: general_correction(
        is_lineal=True,
        build_transformation_function=projective_correction,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.FINDER_CORNERS,
            MatchingFeatures.ALIGNMENTS_CENTERS
        ]
    ),
    Correction.CYLINDRICAL: general_correction(
        is_lineal=False,
        build_transformation_function=cylindrical_transformation,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.FINDER_CORNERS,
            MatchingFeatures.ALIGNMENTS_CENTERS
        ]
    ),
    Correction.TPS: general_correction(
        is_lineal=False,
        build_transformation_function=tps_transformation,
        default_references_features=[
            MatchingFeatures.FINDER_CENTERS,
            MatchingFeatures.FINDER_CORNERS,
            MatchingFeatures.ALIGNMENTS_CENTERS
        ]
    ),
    Correction.DOUBLE_TPS: double_tps
}


def correction(qr: QRCode, method: Optional[Correction] = None, bitpixel: int = 11, border: int = 15,
               **kwargs) -> QRCode:
    """
    Given a QRCode and a selected method of correction applies the correction to the QR Code

    :param qr: A QRCode object
    :param method: A identifier of the method of correction that we want to use
    :param bitpixel: The number of pixels per module that we want in the output of the correction
    :param border: The number of modules of border that we want in the correction
    :param kwargs: Keyword arguments to the selected correction function

    :return: The corrected QRCode object
    """
    if method is None:
        method = Correction.PROJECTIVE

    func = _CORRECTION_METHOD_FUNC[method]
    return func(qr, bitpixel, border, **kwargs)
