import numpy as np


class SplineSpace(object):
    """
    Class for representing a spline space
    """


    def __init__(self, knots, degree):
        """
        Constructor for Spline class. Generates a spline based on given knots,
        coeffs and degree.

        Args:
            knots (np.ndarray):	Knot vector
            degree (int):		Degree of the spline

        Raises:
            TypeError:			If any arg is of the wrong type
        """

        # Check types
        if not isinstance(knots, (np.ndarray)):
            raise TypeError("knot vector must be a numpy array")

        if not isinstance(degree, int):
            raise TypeError("degree must be an integer")

        # Store attributes
        self._knots = knots.astype(np.float64)
        self._degree = degree


    def __len__(self):
        """
        Returns the dimension of the spline space
        """

        return (len(self._knots) - self._degree - 1)


    def __call__(self, x):
        """
        Evaluates all the active basis splines on x

        Args:
            x (float):			Point to evaluate in

        Returns:
            np.ndarray: The value of all active B-splines

        Raises:
            ValueError:			If x is outside the knot vector range
            TypeError:			If any arg is of the wrong type
        """

        return self.evaluate_basis(x)


    def __eq__(self, other):
        """
        Checks wether two spaces are equal or not. Two spline spaces are equal
        if they have the same knot vector, and the same degree.

        Args:
            other (SplineSpace): Space to compare to

        Returns:
            True if spaces are equal, False if not

        Raises:
            TypeError:			If other is not a SplineSpace
        """

        if not isinstance(other, SplineSpace):
            raise TypeError(
                "Cannot compare SplineSpace to %s" % str(type(other)))

        return self._degree == other._degree and (
                self._knots == other._knots).all()


    def find_knot_index(self, x):
        """
        Given a knot vector and a real number x with x in [t_1, t_{n+d+1} ),
        returns the index µ such that t_µ ≤ x < t_{µ+1}.

        Args:
            x (float):			Real value to find the knot interval of

        Returns:
            int: first index µ such that t_µ ≤ x < t_{µ+1}

        Raises:
            ValueError:			If x is outside the knot vector range
            TypeError:			If any arg is of the wrong type
        """

        # Check types
        if x < self._knots[0] or x >= self._knots[-1]:
            raise ValueError(
                "x=%g is outside of knot vector range [%g, %g)"
                % (x, self._knots[0], self._knots[-1])
            )

        if not isinstance(x, (float, int)):
            raise TypeError("x must be a number")

        # Argmax returns first entry when multiple indexes achieve maximum (max
        # is 1 when array is of boolean type)
        return np.argmax(self._knots > x) - 1


    def evaluate_basis(self, x):
        """
        Evaluates all the active basis splines on x

        Args:
            x (float):			Point to evaluate in

        Returns:
            np.ndarray: The value of all active B-splines

        Raises:
            ValueError:			If x is outside the knot vector range
            TypeError:			If any arg is of the wrong type
        """

        # Find knot interval
        mu = self.find_knot_index(x)

        # Initialize b vector
        b = np.zeros(self._degree + 1, dtype=np.float64)
        b[-1] = 1

        for k in range(1, self._degree + 1):
            for j in range(mu - k + 1, mu + 1):
                # Shift index
                i = j - mu - 1

                # Calculate new b values
                b[i - 1] += (self._knots[j + k] - x) / (
                        self._knots[j + k] - self._knots[j]) * b[i]
                b[i] = (x - self._knots[j]) / (
                        self._knots[j + k] - self._knots[j]) * b[i]

        return b


    def get_knots(self):
        """
        Returns a copy of the knot vector for the space that can safely be edited without
        accidentally modifying the space.

        Returns:
            np.ndarray: knot vector for space
        """
        return self._knots.copy()


    def get_degree(self):
        """
        Returns the degree of the space

        Returns:
            int: degree of spline
        """
        return self._degree


    def create_spline(self, coeffs):
        """
        Creates a spline within this spline space with given coefficients

        Args:
            coeffs (np.ndarray): Coefficient vector for the spline

        Returns:
            A Spline object, representing a spline inside the space

        Raises:
            ValueError:			If the number of coefficients doesn't match
                                space dimension.
            TypeError:			If any arg is of the wrong type
        """

        return Spline(self, coeffs)


class Spline(object):
    """
    Class for representing a spline.
    """


    def __init__(self, space: SplineSpace, coeffs):
        """
        Creates a spline within given spline space with given coefficients.
        Consider using create_spline in SplineSpace instead.

        Args:
            space (SplineSpace): Space to create a spline within
            coeffs (np.ndarray): Coefficient vector for the spline

        Raises:
            ValueError:			If the number of coefficients doesn't match
                                space dimension.
            TypeError:			If any arg is of the wrong type
        """

        # Check types
        if not coeffs.T.shape[0] == len(space):
            raise ValueError(
                "Number of coeffs for a spline in a space of degree %d with %d knots must be %d!" % (
                    space._degree, len(space._knots), len(space)))

        if not isinstance(coeffs, (np.ndarray)):
            raise TypeError("coeff vector must be a numpy array")

        if not isinstance(space, SplineSpace):
            raise TypeError("space must be of type SplineSpace")

        # Store attributes
        self._space = space

        if coeffs.ndim == 1:
            self._coeffs = coeffs.astype(np.float64).reshape([1, len(coeffs)])
        else:
            self._coeffs = coeffs.astype(np.float64)


    def __call__(self, x):
        """
        Evaluate a spline in the given point.

        Args:
            x (float):			Point to evaluate spline in

        Returns:
            The value of the spline in the given point

        Raises:
            ValueError:			If x is outside the knot vector range
            TypeError:			If any arg is of the wrong type
        """

        return self.evaluate(x)


    def __add__(self, other):
        """
        Add self to another spline. Ie, if p1 is added to p2, the sum, p3, is
        defined as:

            p3(x) = p1(x) + p2(x)

        for all possible x.

        Args:
            other (Spline):		Spline to add

        Returns:
            A new spline representing the sum of this spline and the other.

        Raises:
            TypeError:			If other is not a spline in the same space.
        """

        if not isinstance(other, Spline):
            raise TypeError("Cannot add Spline to %s" % str(type(other)))

        if not other._space == self._space:
            raise TypeError("Both splines must be from the same spline space")

        return Spline(self._space, self._coeffs + other._coeffs)


    def __eq__(self, other):
        """
        Checks if two splines are equal. Two splines p1 and p2 are equal iff

            p1(x) = p2(x)

        for all possible x.

        Args:
            other (Spline):		Spline to compare to self

        Returns:
            True if spaces are equal, False if not.

        Raises:
            TypeError:			If other is not a Spline.
        """

        if not isinstance(other, Spline):
            raise TypeError("Cannot compare Spline to %s" % str(type(other)))

        return self._space == other._space and (
                self._coeffs == other._coeffs).all()


    def get_coeffs(self):
        """
        Returns a copy of the coefficients to the spline that can safely be edited without
        accidentally modifying the spline.

        Returns:
            np.ndarray: coefficients of spline
        """
        return self._coeffs.copy()


    def get_knots(self):
        """
        Returns a copy of the knot vector for the spline that can safely be edited without
        accidentally modifying the spline.

        Returns:
            np.ndarray: knot vector for spline
        """
        return self._space.get_knots()


    def get_degree(self):
        """
        Returns the degree of the spline

        Returns:
            int: degree of spline
        """
        return self._space.get_degree()


    def is_curve(self):
        """
        Returns True if this object represents a curve, False if it represents a function.

        Returns:
            bool: Whether the Spline object represents a curve or a function
        """
        return self._coeffs.shape[0] != 1


    def get_control_polygon(self):
        """
        Get points for the control polygon of the spline. This will only work for functions
        or 2D curves.

        Returns:
            (np.ndarray, np.ndarray): x and y coordinates for control polygon
        """
        if not self.is_curve():
            ts = np.zeros_like(self._coeffs);

            for j in range(len(self._coeffs)):
                ts[j] = np.sum(self._space._knots[j + 1:j + self._space._degree + 1]) \
                        / self._space._degree

            return ts, self._coeffs.copy()

        else:
            return self._coeffs[0, :].copy(), self._coeffs[1, :].copy()


    def _oslo(self, new_knots, index):
        """
        Implementation of the Oslo algorithm for knot insertion

        Args:
            new_knots:      new knot vector, containing the old as a subset
            index:          index to work at

        Returns:
            new coefficient value for the knot at given index
        """
        # Find knot interval1
        mu = self._space.find_knot_index(new_knots[index])

        # Create some shortcuts because the below expression for c[i] would be
        # crazy ugly otherwise.
        d = self._space._degree
        t = self._space._knots

        # Initialize c vector. Deep copy, so we don't overwrite the
        # coefficients. Ensure float types to avoid integer computations.
        c = self._coeffs[:, mu - d:mu + 1].copy().astype(np.float64)

        for k in range(d, 0, -1):
            for j in range(mu, mu - k, -1):
                # Shift index for indexing in array
                i = j - mu + d

                # Compute next iteration of c_index
                c[:, i] = (new_knots[index + k] - t[j]) / (t[j + k] - t[j]) * c[:, i] \
                          + (t[j + k] - new_knots[index + k]) / (t[j + k] - t[j]) * c[:,
                                                                                    i - 1]

        return c[:, -1]


    def insert_knots(self, new_knots):
        """
        Replace knot vector and recompute coefficients to fit these new knots.

        Args:
            new_knots:      New knot vector containing the old as a subset

        Returns:
            None
        """
        # Create new space object
        new_space = SplineSpace(new_knots, self._space._degree)

        # Initilalize new coeff vector
        additional_knots = len(new_knots) - len(self._space._knots)
        new_coeffs = np.zeros(len(self._coeffs) + additional_knots,
                              dtype=np.float64)

        # Populate new coeff vector
        for i in range(len(new_coeffs)):
            new_coeffs[i] = self._oslo(new_knots, i);

        # Update spline
        self._space = new_space
        self._coeffs = new_coeffs


    def evaluate(self, x):
        """
        Evaluate a spline in the (given) point(s).

        Args:
            x:					Point to evaluate spline in. If x is a float,
                                it will be evaluated in the single point. If it
                                is an array-like object, it will be evaulated in
                                every point.

        Returns:
            float or np.ndarray: The value of the spline in the given point

        Raises:
            ValueError:			If x is outside the knot vector range
            TypeError:			If any arg is of the wrong type
        """
        if isinstance(x, (float, int)):
            return self._inner_evaluate(x)

        elif isinstance(x, (np.ndarray, list, tuple)):
            results = np.zeros([self._coeffs.shape[0], len(x)])

            for i in range(results.shape[1]):
                results[:, i] = self._inner_evaluate(x[i])

            return results

        else:
            raise TypeError("Cannot evaluate type " + type(x))


    def evaluate_all(self):
        """
        Evaluate a spline on entire knot vector

        Returns:
            A tuple of np.ndarrays with arguments and values.
        """
        xs = np.linspace(self._space._knots[0],
                         self._space._knots[-1],
                         100,
                         endpoint=False)
        ps = self.evaluate(xs)

        if not self.is_curve():
            return xs, ps[0, :]
        else:
            return ps


    def _inner_evaluate(self, x):
        """
        Evaluate a spline in the given point.

        Args:
            x (float):			Point to evaluate spline in

        Returns:
            The value of the spline in the given point

        Raises:
            ValueError:			If x is outside the knot vector range
            TypeError:			If any arg is of the wrong type
        """

        # Find knot interval
        mu = self._space.find_knot_index(x)

        # Create some shortcuts because the below expression for c[i] would be
        # crazy ugly otherwise.
        d = self._space._degree
        t = self._space._knots

        # Initialize c vector. Deep copy, so we don't overwrite the
        # coefficients. Ensure float types to avoid integer computations.
        c = self._coeffs[:, mu - d:mu + 1].copy().astype(np.float64)

        for k in range(d, 0, -1):
            for j in range(mu, mu - k, -1):
                # Shift index for indexing in array
                i = j - mu + d

                # Compute next iteration of c_index
                c[:, i] = (x - t[j]) / (t[j + k] - t[j]) * c[:, i] \
                          + (t[j + k] - x) / (t[j + k] - t[j]) * c[:, i - 1]

        return c[:, -1]


class TensorProductSplineSpace(object):

    def __init__(self, spaces):
        self._spaces = spaces


    def create_spline(self, coeffs):
        return SplineSurface(self._spaces, coeffs)


class SplineSurface(object):

    def __init__(self, spaces, coeffs):
        self._spaces = spaces
        self._coeffs = coeffs


### TEST FUNCTIONS:

def _test_find_knot_index():
    knots = np.array([-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5],
                     dtype=np.float64)
    space = SplineSpace(knots, 3)

    assert space.find_knot_index(-0.5) == 3
    assert space.find_knot_index(1.5) == 5
    assert space.find_knot_index(3) == 7
    assert space.find_knot_index(4.5) == 8

    try:
        space.find_knot_index(-2)

        # if we get here, no exception was raised, throw error
        raise AssertionError("No exception raised for x below knot range")

    except ValueError:
        pass

    try:
        space.find_knot_index(6)

        # if we get here, no exception was raised, throw error
        raise AssertionError("No exception raised for x above knot range")

    except ValueError:
        pass


def _test_evaluation(tol=1e-10):
    knots = np.array([-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5],
                     dtype=np.float64)
    coeffs = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
    degree = 3

    space = SplineSpace(knots, degree)

    b0 = np.zeros(100)
    b1 = np.zeros(100)
    b2 = np.zeros(100)
    b3 = np.zeros(100)

    xs = np.linspace(0, 1, 100, endpoint=False)

    for i in range(100):
        b0[i], b1[i], b2[i], b3[i] = space(xs[i])

    # Check that all the basis values sum to 1
    sums = b0 + b1 + b2 + b3
    for s in sums:
        assert (s - 1) < tol

    # Compare the two evaluation methods, they should  yield similar results
    p = space.create_spline(coeffs)
    xs = np.linspace(-1, 5, 100, endpoint=False)
    ps = np.zeros(100)
    ps2 = np.zeros(100)

    for i in range(100):
        ps[i] = p(xs[i])
        mu = space.find_knot_index(xs[i])
        ps2[i] = np.dot(space(xs[i]), coeffs[mu - degree:mu + 1])

    assert np.sum(ps - ps2) < tol


if __name__ == '__main__':
    _test_find_knot_index()
    _test_evaluation()
