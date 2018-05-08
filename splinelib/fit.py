import numpy as np
from .splinelib import *


def least_squares(data, knots, degree, weights=None):
    """
    Fit a non-parametric spline function to the data using least squares optimization

    Args:
        data (np.ndarray):      Data to fit from, with dimensions (m, 2). First column is
                                x direction, and second column is y direction.
        knots (np.ndarray):     Knot vector to fit spline onto
        degree (int):           Desired degree of spline
        weights (np.ndarray):   Optional. Associated weights to data, assumed to all be
                                equal to 1 if omitted. Must be of the same dimensions as
                                data.

    Returns:
        Spline: a spline function fitted to the data
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
    try:
        c = np.linalg.solve(A.T @ A, A.T @ b)
        error = False
    except np.linalg.LinAlgError:
        error = True

    if error:
        raise np.linalg.LinAlgError("No solution to normal equations")

    # Pad with zeros in c corresponding to where we removed columns from A
    c_new = np.zeros_like(non_zero_cols, dtype=np.float64)
    i = 0
    for j in range(len(non_zero_cols)):
        if non_zero_cols[j]:
            c_new[j] = c[i]
            i += 1

    # Return spline using these coeffs
    return space.create_spline(c_new)


def least_squares_3d(data_x, data_y, data_f, knots_x, knots_y, degree, weights_x=None,
                     weights_y=None):
    """
    Fit a non-parametric spline function in two variables to given data using least
    squares optimization.

    Args:
        data_x (np.ndarray):            Data points in x direction. Vector of size m1.
        data_y (np.ndarray):            Data points in y direction. Vector of size m2.
        data_f (np.ndarray):            Estimated function values for each x and y. Matrix
                                        of shape (m1, m2).
        knots_x (np.ndarray):           Knot vector for x space.
        knots_y (np.ndarray):           Knot vector for y space.
        degree (int):                   Degree of spline surface.
        weights_x (np.ndarray):         Weight vector for x direction (of length m1).
                                        Optional, defaults to 1s if omitted.
        weights_y (np.ndarray):         Weight vector for y direction (of length m2).
                                        Optional, defaults to 1s if omitted.

    Returns:
        SplineSurface: A spline surface fitted to the data.
    """
    # Ensure float type to avoid integer computations
    data_x = data_x.astype(np.float64)
    data_y = data_y.astype(np.float64)
    data_f = data_f.astype(np.float64)

    # Default weights to 1 if omitted
    if weights_x is None:
        weights_x = np.ones_like(data_x)

    if weights_y is None:
        weights_y = np.ones_like(data_y)

    # Init space
    space = TensorProductSplineSpace([
        SplineSpace(knots_x, degree),
        SplineSpace(knots_y, degree)
    ])

    # Get dimensions
    m1 = len(data_x)
    m2 = len(data_y)
    n1 = len(knots_x) - degree - 1
    n2 = len(knots_y) - degree - 1

    # Construct data matrices
    A = np.zeros([m1, n1])
    B = np.zeros([m2, n2])
    G = np.zeros([m1, m2])

    space_x, space_y = space.get_spaces()

    # Construct A
    for i in range(m1):
        mu = space_x.find_knot_index(data_x[i])
        A[i, mu - degree:mu + 1] = np.sqrt(weights_x[i]) * space_x(data_x[i])

    # Construct B
    for j in range(m2):
        nu = space_y.find_knot_index(data_y[j])
        B[j, nu - degree:nu + 1] = np.sqrt(weights_y[i]) * space_y(data_y[j])

    # Construct G
    for i, j in np.ndindex(G.shape):
        G[i, j] = np.sqrt(weights_x[i] * weights_y[j]) * data_f[i, j]

    # Solve normal equations
    try:
        D = np.linalg.solve(A.T @ A, A.T @ G)
        C = np.linalg.solve(B.T @ B, B.T @ D.T).T

        error = False
    except np.linalg.LinAlgError:
        error = True

    if error:
        raise np.linalg.LinAlgError("No solution to normal equations")

    return space.create_spline(C)


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
    Fits a parametric curve of arbitrary dimension D from m given data points

    Args:
        data (np.ndarray):              Data to fit curve from. An (m, D) matrix.
        knots (np.ndarray):             Knot vector.
        degree (int):                   Degree of desired spline.
        method (callable):              Optional. Method for approximation. Available
                                        options are
                                            - splinelib.fit.least_squares
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


def fit_surface(data, knots_u, knots_v, degree, method=least_squares_3d,
                parametrization=cord_length, par_u=None, par_v=None):
    """
    Fit a parametric spline surface to given data of arbitrary dimension D.

    Args:
        data (np.ndarray):          A data matrix with shape (n1, n2, D), where n1 and n2
                                    are the number of points in each parameter direction,
                                    and D is the dimension of space (typically 3).
        knots_u (np.ndarray):       Knot vector in u direction.
        knots_v (np.ndarray):       Knot vecotr in v direction.
        degree (int):               Degree of splines.
        method (callable):          Approximation method. Available options are:
                                        - spinelib.fit.least_squares_3d
                                    Optional. Default is least_squares_3d.
        parametrization (callable): Scheme for generation of parametrization. Options are:
                                        - splinelib.fit.uniform
                                        - splinelib.fit.cord_length
                                        - splinelib.fit.centripetal
                                    Optional. Default is cord_length.
        par_u (np.ndarray):         Override parametrization in u direction (this ignores
                                    whatever that is passed as 'parametrization').
        par_v (np.ndarray):         Override parametrization in v direction (this ignores
                                    whatever that is passed as 'parametrization').

    Returns:
        SplineSurface: A parametrix surface fitted to the data.
    """
    # Do parametrization if necessary
    if par_u is None:
        par_u = parametrization(data[:, 0, 0])
    if par_v is None:
        par_v = parametrization(data[0, :, 0])

    # Get implicit parameter values
    D = data.shape[2]
    m1 = len(knots_u) - degree - 1
    m2 = len(knots_v) - degree - 1

    # Initialize result array
    coeffs = np.zeros([m1, m2, D])

    # Fit each dimension using requested method
    for dimension in range(D):
        coeffs[:, :, dimension] = method(
            par_u,
            par_v,
            data[:, :, dimension],
            knots_u,
            knots_v, 3
        ).get_coeffs()

    # Create space and resulting surface.
    space = TensorProductSplineSpace([
        SplineSpace(knots_u, degree),
        SplineSpace(knots_v, degree)
    ])
    return space.create_spline(coeffs)


def generate_uniform_knots(data, degree, length=20, regular=True):
    """
    Generates a d+1-extended knot vector for the data, with knots uniformly distributed.

    Args:
        data (np.ndarray):  Data to fit knots from (parametrization). Used to find min and
                            max value needed.
        degree (int):       Degree of spline to fit on knot vector.
        length (int):       Optional. Length of created knot vector. Default is 20.
        regular (bool):     Optional. Make vector d+1-regular in addition to d+1-extended.
                            Default is True

    Returns:
        np.ndarray: A knot vector for the data with knots uniformly distributed.
    """
    # Find min and max, and store something slightly less/bigger
    minval = np.min(data)
    maxval = np.max(data) + 1e-12

    # Create inner knots, each with a multiplicity of 1
    inner_knots = np.linspace(minval, maxval, length - 2 * (degree), endpoint=True)

    if regular:
        # Pad with d+1 entries of min/maxval at the ends to make the knot d+1-regular
        knots = np.zeros(2 * degree + len(inner_knots))
        knots[0:degree] = degree * [minval]
        knots[degree:len(inner_knots) + degree] = inner_knots
        knots[len(inner_knots) + degree:] = degree * [maxval]

    else:
        # Find step size and continue d+1 times in each direction
        step_size = inner_knots[1] - inner_knots[0]

        knots = np.linspace(minval - (degree) * step_size,
                            maxval + (degree) * step_size,
                            length,
                            endpoint=True)

    # Shift knots with something very small to cope with rounding errors (knot indices
    # will depend on stuff being strictly smaller).
    return knots - 1e-13


def sample(spline, num=20, min=None, max=None):
    """
    Samples the spline uniformly, and returns points.

    Args:
        spline (Spline):    Spline to sample
        num (int):          Number of points to sample. Optional, defaults to 20.
        min (int):          Lower bound of sampling range. Optional, defaults to lower
                            bound of the support for the spline if omitted.
        max (int):          Upper bound of sampling range. Optional, defaults to upper
                            bound of the support for the spline if omitted.

    Returns (np.ndarray): Sampled points

    """
    if min is None:
        min = spline.get_support()[0]
    if max is None:
        max = spline.get_support()[1]

    xs = np.linspace(min, max, num, endpoint=False)

    return spline(xs)
