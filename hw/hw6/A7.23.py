import numpy as np
import matplotlib.pyplot as plt
from disks_data import *
import cvxpy as cvx

c = cvx.Variable((n, 2))
r = cvx.Variable(n)

min_area_obj = cvx.Minimize(cvx.sum_squares(r))
min_perim_obj = cvx.Minimize(cvx.sum(r))

constraints = [r >= 0, c[:k, :] == Cgiven[:k, :], r[:k] == Rgiven[:k]]

for i in range(len(Gindexes)):
	constraints += [cvx.norm(c[Gindexes[i, 0], :] - c[Gindexes[i, 1], :])\
	<= (r[Gindexes[i, 0]] + r[Gindexes[i, 1]])]

min_total_area = cvx.Problem(min_area_obj, constraints)
min_total_perim = cvx.Problem(min_perim_obj, constraints)

opt_area = min_total_area.solve(solver = cvx.CVXOPT)
print("optimal area: {}".format(np.pi * opt_area))
plot_disks(c.value, r.value, Gindexes, 'area.png')

opt_perim = min_total_perim.solve(solver = cvx.CVXOPT)
print("optimal perimeter: {}".format(2 * np.pi * opt_perim))
plot_disks(c.value, r.value, Gindexes, 'perim.png')
