import numpy as np
from .splinelib import *


def least_squares(data, knots, degree, weights=None):
    """
    Fit a spline to the data using least squares optimization

    Args:
        data (np.ndarray):      Data to fit from, with dimensions (m, 2). First column is
                                x direction, and second column is y direction.
        knots (np.ndarray):     Knot vector to fit spline onto
        degree (int):           Desired degree of spline
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

    # Mask out zero columns so we avoid a singular matrix in the normal equations
    non_zero_cols = np.sum(A, axis=0) > 0.001
    A = A[:, non_zero_cols]

    b = np.sqrt(weights) * data[:, 1]

    # Solve normal equations
    c = np.linalg.solve(A.T @ A, A.T @ b)

    # Pad with zeros in c corresponding to where we removed columns from A
    c_new = np.zeros_like(non_zero_cols, dtype=np.float64)
    i = 0
    for j in range(len(non_zero_cols)):
        if non_zero_cols[j]:
            c_new[j] = c[i]
            i += 1

    # Return spline using these coeffs
    return space.create_spline(c_new)


def uniform(data):
    """
    Returns a column vector with parametrization for the data using the uniform
    parametrization scheme.

    Args:
        data (np.ndarray):         Points to parametrize

    Returns:
        np.ndarray: Uniform parametrization of the data
    """
    return np.arange(0, data.shape[0]).reshape([data.shape[0], 1])


def cord_length(data):
    """
    Returns a column vector with parametrization for the data using the cord length
    parametrization scheme.

    Args:
        data (np.ndarray):         Points to parametrize

    Returns:
        np.ndarray: Cord length parametrization of the data
    """
    u = np.zeros(data.shape[0])

    for i in range(1, len(u)):
        u[i] = u[i - 1] + np.linalg.norm(data[i, :] - data[i - 1, :], ord=2)

    return u.reshape([len(u), 1])


def centripetal(data):
    """
    Returns a column vector with parametrization for the data using the centripetal
    parametrization scheme.

    Args:
        data (np.ndarray):         Points to parametrize

    Returns:
        np.ndarray: Centripetal parametrization of the data
    """
    u = np.zeros(data.shape[0])

    for i in range(1, len(u)):
        u[i] = u[i - 1] + np.sqrt(np.linalg.norm(data[i, :] - data[i - 1, :], ord=2))

    return u.reshape([len(u), 1])


def fit_curve(data, knots, degree, method=least_squares, parametrization=cord_length):
    """
    Fits a curve of arbitrary dimension D from m given data points

    Args:
        data (np.ndarray):              Data to fit curve from. A (m, D+1) matrix. First
                                        column is the parametrization,
        knots (np.ndarray):             Knot vector
        degree (int):                   Degree of desired spline
        method (callable):              Optional. Method for approximation. Available
                                        options are
                                            - splinelib.fit.least_squares.
        parametrization (callable):     Optional. Method for parametrization. Available
                                        options are
                                            - splinelib.fit.uniform
                                            - splinelib.fit.cord_length
                                            - splinelib.fit.centripetal
                                        Default is cord_length.

    Returns:
        Spline: A spline curve fitted to the data.
    """
    D = data.shape[1]
    coeffs = np.zeros([D, len(knots) - degree - 1])

    # Generate parametrization and augment data
    par = parametrization(data)
    data = np.hstack([par, data])

    # Approximate coeffs in each dimension
    for dimension in range(D):
        subset = np.vstack([data[:, 0], data[:, dimension + 1]]).T
        coeffs[dimension, :] = method(subset, knots, degree).get_coeffs()

    # Create spline and return
    space = SplineSpace(knots, degree)
    return space.create_spline(coeffs)


def generate_uniform_knots(data, degree, step_size=1):
    """
    Generates a d+1-regular knot vector for the data, with knots uniformly distributed.

    Args:
        data:           Data to fit knots from (parametrization). Used to find min and max
                        value needed.
        degree:         Degree of spline to fit on knot vector.
        step_size:      Optional. Step size between entries in the knot vector. Default is 1.

    Returns:

    """
    # Find min and max, and store something slightly less/bigger
    minval = np.floor(np.min(data) * 10) / 10
    maxval = np.ceil(np.max(data) * 10) / 10

    # Create inner knots, each with a multiplicity of 1
    inner_knots = np.arange(minval + 1, maxval, step_size)

    # Pad with d+1 entries of min/maxval at the ends to make the knot d+1-regular
    knots = np.zeros(2 * (degree + 1) + len(inner_knots))
    knots[0:degree + 1] = (degree + 1) * [minval]
    knots[degree + 1:len(inner_knots) + degree + 1] = inner_knots
    knots[len(inner_knots) + degree + 1:] = (degree + 1) * [maxval]

    return knots

