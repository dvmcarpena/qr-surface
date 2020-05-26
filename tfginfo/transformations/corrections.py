from typing import Callable, Optional

import numpy as np
from skimage import transform

from tfginfo.qr import QRCode, Correction
from tfginfo.utils import rgb2binary2rgb
from tfginfo.matching import MatchingFeatures

from .interpolate import RadialBasisSplines
from .general import general_correction
from .geometry import LSqEllipse, project_to_cylinder

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
    f1 = qr.finder_patterns[0]
    f2 = qr.finder_patterns[1]
    f3 = qr.finder_patterns[2]

    max_side = max(f1.corners[0][0] - f2.corners[1][0], f1.corners[0][1] - f3.corners[3][1])
    sign = np.sign(
        (f1.corners[0][1] + f2.corners[1][1])
        - (f1.corners[1][1] + f2.corners[0][1])
    )
    ellipse_radius = sign * int(2 * max_side / 3)

    ellipse_points = np.concatenate((
        f1.corners[0][::-1].reshape((1, 2)),
        f1.corners[1][::-1].reshape((1, 2)),
        f2.corners[0][::-1].reshape((1, 2)),
        f2.corners[1][::-1].reshape((1, 2))
    ), axis=0)

    ellipse_points_2 = ellipse_points.copy()
    ellipse_points_2[:, 1] = 2 * ellipse_radius - ellipse_points_2[:, 1]
    ellipse_points = np.concatenate((
        ellipse_points,
        ellipse_points_2
    ))

    ellipse = LSqEllipse()
    ellipse.fit([ellipse_points[:, 0], ellipse_points[:, 1]])
    ellipse_center, ellipse_width, height, phi = ellipse.parameters()

    # from matplotlib import pyplot as plt
    # from matplotlib.patches import Ellipse
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # fig = plt.figure(figsize=[24, 12])
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_title("Original Image")
    #
    # ellipse_patch = Ellipse(xy=ellipse_center, width=2 * ellipse_width, height=2 * height, angle=np.rad2deg(phi),
    #                         edgecolor='g', fc='None', lw=2, label='Fit', zorder=2)
    # ax.add_patch(ellipse_patch)
    # # ellipse_patch = Ellipse(
    # #     xy=(
    # #         ellipse_center[0] - old_sq[0][1][0] + old_sq[2][2][0],
    # #         ellipse_center[1] - old_sq[0][1][1] + old_sq[2][2][1]
    # #     ),
    # #     width=2 * ellipse_width + 30,
    # #     height=2 * height,
    # #     angle=np.rad2deg(phi),
    # #     edgecolor='r', fc='None', lw=2, label='Fit', zorder=2)
    # # ax.add_patch(ellipse_patch)
    # ax.imshow(qr.image)
    # ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'ro', label='test data', zorder=1)
    # ax.plot(src[:, 0], src[:, 1], 'ro', label='test data', zorder=1)
    # plt.show()

    cylinder_center_x = ellipse_center[0]
    cylinder_radius = ellipse_width
    cylinder_dst = project_to_cylinder(dst, cylinder_center_x, cylinder_radius)

    # n, m, i = 2, 2, 1
    # fig = plt.figure(figsize=[24, 12])
    #
    # ax: Axes3D = fig.add_subplot(n, m, i, projection='3d')
    # ax.set_title("Destiny result points")
    # ax.scatter(dst[:, 0], dst[:, 1], np.full_like(dst[:, 1], 0), c="red")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_zlim(0, 800)
    # i += 1
    #
    # ax: Axes3D = fig.add_subplot(n, m, i, projection='3d')
    # ax.set_title("Cilindric projection of ideal points")
    # ax.scatter(cylinder_dst[:, 0], cylinder_dst[:, 1], cylinder_dst[:, 2], c="blue")
    # ax.scatter(dst[:, 0], dst[:, 1], np.full_like(dst[:, 1], 0), c="red")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_zlim(0, 800)
    # i += 1
    #
    # ax: Axes3D = fig.add_subplot(n, m, i, projection='3d')
    # ax.set_title("Source image points")
    # ax.scatter(src[:, ::-1][:, 0], src[:, ::-1][:, 1], np.full_like(src[:, 1], 0), c="green", marker="*")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_zlim(0, 800)
    # i += 1
    #
    # ax: Axes3D = fig.add_subplot(n, m, i, projection='3d')
    # ax.set_title("All in one plot")
    # ax.scatter(cylinder_dst[:, 0], cylinder_dst[:, 1], cylinder_dst[:, 2], c="blue")
    # ax.scatter(dst[:, 0], dst[:, 1], np.full_like(dst[:, 1], 0), c="red")
    # ax.scatter(src[:, ::-1][:, 0], src[:, ::-1][:, 1], np.full_like(src[:, 1], 0), c="green", marker="*")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_zlim(0, 800)
    # i += 1

    b_u = []
    b_v = []
    d_u = []
    d_v = []
    for (x, y, z), (u, v) in zip(cylinder_dst, src[:, ::-1]):
        # X Y Z 1 0 0 0 0 −uX −uY −uZ
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

    xy = np.array(np.meshgrid(np.arange(dst_size),np.arange(dst_size))).T.reshape(-1, 2)
    cylinder_dst = project_to_cylinder(xy, cylinder_center_x, cylinder_radius)
    cylinder_dst = np.hstack((cylinder_dst, np.ones([cylinder_dst.shape[0], 1])))
    UV = matrix @ cylinder_dst.T
    UV /= UV[2]
    return lambda _: UV[:2].T


def tps_transformation(_: QRCode, src: np.ndarray, dst: np.ndarray, dst_size: int):
    tps_inv_tps_m = RadialBasisSplines(
        source_landmarks=dst[:, ::-1],
        destiny_landmarks=src[:, ::-1],
        function="thin_plate",
        norm="euclidean",
        # smooth=1
    )

    markers = np.stack(np.meshgrid(np.arange(dst_size), np.arange(dst_size)), axis=2)
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
            # MatchingFeatures.FOURTH_CORNER
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
