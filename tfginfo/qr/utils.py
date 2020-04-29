import numpy as np
from scipy import spatial

from tfginfo.utils import Array


def guess_version(version_points: Array) -> int:
    """
    Algorithm that find the version of a QR code using 2 outer corners of 2 of the
    position patters, that need to be on the same side of the QR, and that side
    needs to be close to being straight.

    :param version_points: Array of 2 outer corners of 2 of the position patters.

    :return: The version guessed from the points given.
    """
    dists = spatial.distance_matrix(version_points, version_points, p=2)
    cross_ratio = (dists[0, 2] * dists[1, 3]) / (dists[0, 3] * dists[1, 2])

    x = 7 * (np.sqrt(cross_ratio / (cross_ratio - 1)) - 1)
    return int(np.round((x - 3) / 4))
