import logging

import numpy as np
from scipy.interpolate import Rbf

logger = logging.getLogger(__name__)


class RadialBasisSplines(Rbf):
    def __init__(self, source_landmarks, destiny_landmarks, *args, **kwargs):
        self.src = source_landmarks
        self.dst = destiny_landmarks

        self._a = None
        self._p = None
        self._y = None
        self._l = None
        self._w = None

        """
        Call super() to compute the kernel of the splines and also the radial basis function, 
        doesn't matter the "dst" array now.
        w_k == self.A
        biharmonic_spline == self._function
        """
        args = [*source_landmarks.T, self.dst[:, 0]]
        super(RadialBasisSplines, self).__init__(
            *args,
            function=kwargs.pop("function", "linear"),
            smooth=kwargs.pop("smooth", 0.0),
            norm=kwargs.pop("norm", None),
            epsilon=kwargs.pop("epsilon", None)
        )

    @property
    def A(self):
        if self._a is None:
            self._a = super(RadialBasisSplines, self).A
        return self._a

    @staticmethod
    def _expand_P(w_p):
        n0, *_ = w_p.shape
        ones = np.ones((n0, 1))
        return np.concatenate((ones, w_p), axis=1)

    @property
    def P(self):
        if self._p is None:
            self._p = self._expand_P(self.src)
        return self._p

    @staticmethod
    def _expand_Y(dst):
        _, m, *_ = dst.shape
        return np.concatenate((dst, np.zeros((m + 1, m))))

    @property
    def Y(self):
        if self._y is None:
            self._y = self._expand_Y(self.dst)
        return self._y

    @staticmethod
    def _compute_L(w_k, w_p):
        _, m, *_ = w_p.shape
        w_l_down = np.concatenate((w_p.T, np.zeros((m, m))), axis=1)
        w_l_up = np.concatenate((w_k, w_p), axis=1)
        return np.concatenate((w_l_up, w_l_down))

    @property
    def L(self):
        # Note self.A is matrix K in Principials Warps (Bookstein):
        if self._l is None:
            self._l = self._compute_L(self.A, self.P)
        return self._l

    def _compute_W(self):
        return np.linalg.solve(self.L, self.Y)
        # return np.dot(np.linalg.inv(w_l), w_y)

    @property
    def W(self):
        if self._w is None:
            self._w = self._compute_W()
        return self._w

    @property
    def W_W(self):
        return self.W[:self.N]

    @property
    def W_A(self):
        return self.W[self.N:]

    def _compute_K(self, xa):
        n0, *_ = xa.shape
        n1, *_ = self.src.shape

        rep_mat0 = np.tile(xa, (n1, 1, 1))
        rep_mat1 = np.tile(self.src, (n0, 1, 1))

        # TODO use self.norm in this point
        w_r = np.sqrt(np.power((rep_mat0 - rep_mat1.transpose(1, 0, 2)), 2).sum(axis=2))
        return self._function(w_r)

    def _t_default(self, xa):
        w_p = self._expand_P(xa)
        w_k = self._compute_K(xa)
        w_l = np.concatenate((w_k, w_p.T))
        return np.dot(w_l.T, self.W)

    def _t_affine(self, xa):
        translation = self.W_A[0]
        rotation = self.W_A[1:].transpose()
        return np.dot(rotation, xa.transpose()).transpose() + translation

    def _t_affine_free(self, xa):
        w_k = self._compute_K(xa)
        return np.dot(w_k.T, self.W_W)

    def _t_radial_basis(self, xa):
        w_k = self._compute_K(xa)
        return np.dot(w_k.T, np.linalg.solve(self.A, self.dst))

    def _call_norm(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1[np.newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[np.newaxis, :]
        x1 = x1[..., :, np.newaxis]
        x2 = x2[..., np.newaxis, :]
        return self.norm(x1.T, x2.T)  # TODO reintegrate with super()._call_norm (avoid .T)

    def __call__(self, *args, **kwargs):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        xa = np.asarray(args, dtype=np.float_).transpose()

        method = kwargs.pop("method", "default").replace("-", "_").replace(" ", "_")
        try:
            method = getattr(self, f"_t_{method}")
        except AttributeError:
            available_methods = [x[3:] for x in dir(self) if x.startswith('_t_')]
            raise ValueError(f"Method must be one of {', '.join(available_methods)}")

        return method(xa)
