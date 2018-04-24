import numpy as np
import matplotlib.pyplot as plt
from splinelib import *
from plotting import *


def least_squares(knots, degree, data, weights=None):
    """
    Fit a spline to the data using least squares optimization

    Args:
        knots (np.ndarray):     Knot vector to fit spline onto
        degree (int):           Desired degree of spline
        data (np.ndarray):      Data to fit from, with dimensions (m, 2). First column is
                                x direction, and second column is y direction.
        weights (np.ndarray):   Optional. Associated weights to data, assumed to all be
                                equal to 1 if omitted. Must be of the same dimensions as
                                data.

    Returns:
        Spline: a spline fitted to the data
    """
    # Ensure float type to avoid integer computations
    data = data.astype(np.float64)

    # Init space
    space = SplineSpace(knots, degree)

    # Get dimensions
    m, _ = data.shape
    n = len(knots) - degree - 1

    # Initialize weights to 1 if arg is omitted
    if weights is None:
        weights = np.ones(m, dtype=np.float64)

    # Construct data matrix and vector
    A = np.zeros([m, n], dtype=np.float64)
    for i in range(m):
        mu = space.find_knot_index(data[i, 0])
        A[i, mu - degree:mu + 1] = np.sqrt(weights[i]) * space(data[i, 0])

    b = np.sqrt(weights) * data[:, 1]

    # Solve normal equations and return spline using these coeffs
    return space.create_spline(np.linalg.solve(A.T @ A, A.T @ b))


def uniform(data):
    return np.arange(0, data.shape[0])


def cord_length(data):
    u = np.zeros(data.shape[0])
    for i in range(1, len(u())):
        u[i] = u[i - 1] + np.linalg.norm(data[i, :] - data[i - 1, :], ord=2)
    return u


def centripetal(data):
    u = np.zeros(data.shape[0])
    for i in range(1, len(u())):
        u[i] = u[i - 1] + np.sqrt(np.linalg.norm(data[i, :] - data[i - 1, :], ord=2))
    return u


def fit_curve(data, degree, knots, method=least_squares):
    """
    Fits a curve of arbitrary dimension D from m given data points

    Args:
        data (np.ndarray):              Data to fit curve from. A (m, D) matrix.
        degree (int):                   Degree of desired spline
        knots (np.ndarray):             Knot vector
        method (callable):              Optional. Method for approximation. Available
                                        options are splinelib.fit.least_squares.

    Returns:
        Curve: A spline curve fitted to the data.
    """
    pass



def _test_fit():
    data = np.loadtxt("hj1.dat")

    print(data[:,0:2])

    curve = fit_curve(data, 3, )


if __name__ == "__main__":
    _test_fit()
