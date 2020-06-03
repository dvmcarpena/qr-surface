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


class LSqEllipse:
    """
    Demonstration of least-squares fitting of ellipses

    __author__ = "Ben Hammel, Nick Sullivan-Molina"
    __credits__ = ["Ben Hammel", "Nick Sullivan-Molina"]
    __maintainer__ = "Ben Hammel"
    __email__ = "bdhammel@gmail.com"
    __status__ = "Development"

    References:
    (*) Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares
        Fitting of Ellipses'
    (**) http://mathworld.wolfram.com/Ellipse.html
    (***) White, A. McHale, B. 'Faraday rotation data analysis with least-squares
        elliptical fitting'

    """

    def __init__(self):
        self.coef = None
        self._center = None
        self._width = None
        self._height = None
        self._phi = None

    def fit(self, data):
        """
        Least Squares fitting algorithm. Theory taken from (*). Solving equation:
            Sa=lCa,
        with
            a = |a b c d f g>
            a1 = |a b c>
            a2 = |d f g>

        :param data: List of two lists containing the x and y data of the
            ellipse of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]

        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        data = np.array(data, dtype=float)
        x, y = data[0], data[1]

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T

        # Linear part of design matrix [eqn. 16] from (*)
        D2 = np.vstack([x, y, np.ones(len(x))]).T

        # Forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        # Reduced scatter matrix [eqn. 29]
        M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        _, evec = np.linalg.eig(M)

        # Eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * evec[0, :] * evec[2, :] - np.power(evec[1, :], 2)
        a1 = evec[:, np.nonzero(cond > 0)[0]]

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -np.linalg.inv(S3) @ S2.T @ a1

        # Eigenvectors |a b c d f g>
        self.coef = np.vstack([a1, a2])
        self._save_parameters()

    def _save_parameters(self):
        """
        Finds the important parameters of the fitted ellipse. Theory taken form http://mathworld.wolfram
        """
        # Eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0, 0]
        b = self.coef[1, 0] / 2.
        c = self.coef[2, 0]
        d = self.coef[3, 0] / 2.
        f = self.coef[4, 0] / 2.
        g = self.coef[5, 0]

        # Finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        denominator1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        denominator2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        width = np.sqrt(numerator / denominator1)
        height = np.sqrt(numerator / denominator2)

        # Angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5 * np.arctan((2. * b) / (a - c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """
        Angle of counterclockwise rotation of major-axis of ellipse to x-axis. [eqn. 23] from (**).
        """
        return self._phi

    def parameters(self):
        return self.center, self.width, self.height, self.phi
