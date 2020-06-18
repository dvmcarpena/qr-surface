import numpy as np


def collinear(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Given three points checks if they are collinear

    :param p0: The first 2D point
    :param p1: The second 2D point
    :param p2: The third 2D point

    :return: A boolean that tells if they are collinear or not
    """
    det = np.linalg.det(np.array([
        [1, p0[0], p0[1]],
        [1, p1[0], p1[1]],
        [1, p2[0], p2[1]]
    ]))
    cot = max(np.linalg.norm(p0), np.linalg.norm(p1), np.linalg.norm(p2))
    return abs(det) < cot


def project_to_cylinder(xy: np.ndarray, cil_cen: float, cil_rad: float) -> np.ndarray:
    """
    Projects the given 2D points in a 3D cylinder of axis center of coordinates
    (center_x, :, 0), and radius given

    :param xy: 2D points to project to the cylinder
    :param cil_cen: First coordinate of the cylinder center axis
    :param cil_rad: Radius of the cylinder

    :return: 3D points projections of the ones given
    """
    try:
        z = np.sqrt(cil_rad ** 2 - np.power(cil_cen - xy[:, 0], 2))
    except RuntimeWarning:
        z = np.zeros_like(xy[:, 0])
    return np.vstack((xy[:, 0], xy[:, 1], z)).T
