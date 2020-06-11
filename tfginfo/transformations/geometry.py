import numpy as np


def project_to_cylinder(xy: np.ndarray, center_x: float, radius: float, naxis: int = 0) -> np.ndarray:
    """
    Projects the given 2D points in a 3D cylinder of axis center of coordinates
    (center_x, :, 0), and radius given.

    :param xy: 2D points to project to the cylinder.
    :param center_x: First coordinate of the cylinder center axis.
    :param radius: Radius of the cylinder.

    :return: 3D points projections of the ones given.
    """
    # print(radius ** 2 - np.power(center_x - xy[:, naxis], 2))
    # z = np.sqrt(radius ** 2 - np.power(center_x - xy[:, naxis], 2))
    z = np.sqrt(radius ** 2 - np.power(center_x - xy[:, naxis], 2))
    return np.vstack((xy[:, 0], xy[:, 1], z)).T
