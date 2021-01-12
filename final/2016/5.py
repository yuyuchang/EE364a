import numpy as np
import cvxpy as cp

from ideal_pref_point_data import *

c_ideal_min = cp.Variable(n)
c_ideal_max = cp.Variable(n)

# Solve ideal min
for idx in range(n):
	obj = cp.Minimize(c_ideal_min[idx])
	constraints = [c_ideal_min >= 0, c_ideal_min <= 1]
	for d_ in d:
		i = d_[0]
		j = d_[1]
		constraints += [(c[j] - c[i]).T @ c_ideal_min <= 0.5 * (c[i] + c[j]).T @ (c[j] - c[i])]
	prob = cp.Problem(obj, constraints)
	prob.solve()
	print(prob.value)

