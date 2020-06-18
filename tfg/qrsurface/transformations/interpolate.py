from typing import TypeVar

import numpy as np
from scipy.special import xlogy
from scipy.spatial import distance

N = TypeVar('N', float, np.ndarray)


def thin_plate_rbf(r: N) -> N:
    """
    Radial basis function of the Thin Plate Splines in 2 dimensions

    :param r: A positive real value or array of them

    :return: The thin plate RBF of r, with the same type and shape as r
    """
    return xlogy(r ** 2, r)


class ThinPlateSpline:
    """
    A Thin Plate Spline approximation, which is constructed by two sets of source and destiny landmarks.
    The application of the interpolated transformation can be applied to one or more points using the call method.
    """

    def __init__(self, source_landmarks: np.ndarray, destiny_landmarks: np.ndarray, smooth: float = 0.0) -> None:
        self.src = source_landmarks
        self.dst = destiny_landmarks
        self.smooth = smooth

        d, n = self.src.shape
        m = self.dst.shape[1]

        r = distance.squareform(distance.pdist(self.src, 'euclidean'))
        self.K = thin_plate_rbf(r) - np.eye(self.src.shape[0]) * self.smooth
        self.P = np.concatenate((np.ones((d, 1)), self.src), axis=1)
        self.Y = np.concatenate((self.dst, np.zeros((m + 1, m))))

        w_l_down = np.concatenate((self.P.T, np.zeros((n + 1, n + 1))), axis=1)
        w_l_up = np.concatenate((self.K, self.P), axis=1)
        self.L = np.concatenate((w_l_up, w_l_down))

        self.WA = np.linalg.solve(self.L, self.Y)

    def __call__(self, xa: np.ndarray) -> np.ndarray:
        w_p = np.concatenate((np.ones((xa.shape[0], 1)), xa), axis=1)
        rep_mat0 = np.tile(xa, (self.src.shape[0], 1, 1))
        rep_mat1 = np.tile(self.src, (xa.shape[0], 1, 1))
        w_r = np.sqrt(np.power((rep_mat0 - rep_mat1.transpose((1, 0, 2))), 2).sum(axis=2))
        w_k = thin_plate_rbf(w_r)

        w_l = np.concatenate((w_k, w_p.T))
        return np.dot(w_l.T, self.WA)
