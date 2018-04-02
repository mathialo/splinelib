import numpy as np
import matplotlib.pyplot as plt
from splinelib import *

def add_spline_to_plot(spline):
	xs, ps = spline.evaluate_all()
	plt.plot(xs, ps)
	plt.plot(*spline.get_control_polygon())

	min_val = np.min(ps)
	dist = np.max(ps) - min_val

	steps = [i * (dist/8) - dist for i in range(spline._space._degree + 1)]
	similar_knots = 0
	previous_knot = None

	def double_equals(d1, d2, tol=1e-10):
		if d1 is None or d2 is None: return False
		else: return abs(d1 - d2) < tol

	for knot in spline._space._knots:
		if double_equals(knot, previous_knot):
			similar_knots += 1
		else:
			similar_knots = 0

		plt.scatter(knot, steps[similar_knots], color="k", s=10)

		previous_knot = knot


