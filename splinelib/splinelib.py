import numpy as np


class SplineSpace(object):
	"""
	Class for representing a spline space
	"""

	def __init__(self, knots: np.ndarray, degree: int):
		"""
		Constructor for Spline class. Generates a spline based on given knots, 
		coeffs and degree. 

		Input:
			knots (np.ndarray):	Knot vector
			degree (int):		Degree of the spline

		Returns:
			A spline object

		Raises:
			TypeError:			If any arg is of the wrong type
		"""

		super(SplineSpace, self).__init__()

		# Check types
		if not isinstance(knots, (np.ndarray)):
			raise TypeError("knot vector must be a numpy array")

		if not isinstance(degree, int):
			raise TypeError("degree must be an integer")

		# Store attributes
		self._knots = knots
		self._degree = degree


	def __len__(self) -> int:
		"""
		Returns the dimension of the spline space
		"""

		return (len(self._knots) - self._degree - 1)


	def __call__(self, x: float) -> np.ndarray:
		"""
		Evaluates all the active basis splines on x

		Input:
			x (float):			Point to evaluate in

		Returns:
			The value of all active B-splines

		Raises:
			ValueError:			If x is outside the knot vector range
			TypeError:			If any arg is of the wrong type
		"""

		return self.evaluate_basis(x)


	# Exercise 2.17
	def find_knot_index(self, x: float) -> int:
		"""
		Given a knot vector and a real number x with x in [t_1, t_{n+d+1} ), 
		returns the index µ such that t_µ ≤ x < t_{µ+1}.

		Inputs:
			x (float):			Real value to find the knot interval of

		Returns:
			first index µ such that t_µ ≤ x < t_{µ+1}

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


	# Algorithm 2.17
	def evaluate_basis(self, x: float) -> np.ndarray:
		"""
		Evaluates all the active basis splines on x

		Input:
			x (float):			Point to evaluate in

		Returns:
			The value of all active B-splines

		Raises:
			ValueError:			If x is outside the knot vector range
			TypeError:			If any arg is of the wrong type
		"""

		# Find knot interval
		mu = self.find_knot_index(x)

		# Initialize b vector
		b = np.zeros(self._degree+1, dtype=np.float64)
		b[-1] = 1

		for k in range(1, self._degree+1):
			for j in range(mu-k+1, mu+1):
				# Shift index
				i = j-mu-1

				# Calculate new b values
				b[i-1] += (self._knots[j+k] - x) / (self._knots[j+k] - self._knots[j])*b[i]
				b[i]    =  (x - self._knots[j])  / (self._knots[j+k] - self._knots[j])*b[i]

		return b


	def create_spline(self, coeffs: np.ndarray):
		"""
		Creates a spline within this spline space with given coefficients

		Input:
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

	def __init__(self, space: SplineSpace, coeffs: np.ndarray):
		"""
		Creates a spline within given spline space with given coefficients.
		Consider using create_spline in SplineSpace instead.

		Input:
			space (SplineSpace): Space to create a spline within
			coeffs (np.ndarray): Coefficient vector for the spline

		Returns:
			A Spline object, representing a spline inside the space

		Raises:
			ValueError:			If the number of coefficients doesn't match 
								space dimension. 
			TypeError:			If any arg is of the wrong type
		"""

		super(Spline, self).__init__()

		# Check types
		if not len(coeffs) == len(space):
			raise ValueError("Number of coeffs for a spline in a space of degree %d with %d knots must be %d!" % (space._degree, len(space._knots), len(space)))

		if not isinstance(coeffs, (np.ndarray)):
			raise TypeError("coeff vector must be a numpy array")

		if not isinstance(space, SplineSpace):
			raise TypeError("space must be of type SplineSpace")

		# Store attributes
		self._space = space
		self._coeffs = coeffs


	def __call__(self, x: float) -> float:
		"""
		Evaluate a spline in the given point.

		Input:
			x (float):			Point to evaluate spline in

		Returns:
			The value of the spline in the given point

		Raises:
			ValueError:			If x is outside the knot vector range
			TypeError:			If any arg is of the wrong type
		"""

		return self.evaluate(x)


	# Algorithm 2.16
	def evaluate(self, x: float) -> float:
		"""
		Evaluate a spline in the given point.

		Input:
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
		c = self._coeffs[mu-d:mu+1].copy().astype(np.float64)

		for k in range(d, 0, -1):
			for j in range(mu, mu-k, -1):
				# Shift index for indexing in array
				i = j-mu+d

				# Compute next iteration of c_index
				c[i] = (x - t[j]) / (t[j+k] - t[j]) * c[i] + (t[j+k] - x) / (t[j+k] - t[j]) * c[i-1]

		return c[-1]



### TEST FUNCTIONS:

def _test_find_knot_index():
	knots = np.array([-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5], dtype = np.float64)
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
	knots = np.array([-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5], dtype = np.float64)
	coeffs = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1], dtype = np.float64)
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
		ps2[i] = np.dot(space(xs[i]), coeffs[mu-degree:mu+1])

	assert np.sum(ps - ps2) < tol


if __name__ == '__main__':
	_test_find_knot_index()
	_test_evaluation()
