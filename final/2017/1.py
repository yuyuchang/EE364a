import numpy as np
import scipy as sc
import cvxpy as cp

import matplotlib.pyplot as plt
from scipy.stats import norm as normal
from transform_to_normal_data import *

Dmax = 0.05
z = np.zeros(n, )
e = np.ones(n, )
lb_const = np.array(range(1, n + 1)) * 1.0 / n
lb_Dmax = lb_const - Dmax
ub_const = np.array(range(0, n)) * 1.0 / n
ub_Dmax = ub_const + Dmax
lb = normal.ppf(np.amax(np.column_stack((lb_Dmax, z)), axis = 1))
ub = normal.ppf(np.amin(np.column_stack((ub_Dmax, e)), axis = 1))

lb_idx = (lb != -np.Inf)
ub_idx = (ub != np.Inf)

lb = lb[lb_idx]
ub = ub[ub_idx]

y = cp.Variable(n)
obj = 0
for i in range(n - 2):
	obj += cp.abs((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i]))

obj = cp.Minimize(obj)
constraints = [lb <= y[lb_idx], y[ub_idx] <= ub]
prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.ECOS)

print(prob.value)
y = np.squeeze(np.array(y.value))

xl = np.amin(x) - 1
xr = np.amax(x) + 0.1
yl = np.amin(y) - 1
yr = np.amax(y) + 0.1
xnormp = np.linspace(xl, xr, num = 1000)
px = normal.cdf(xnormp)
ynormp = np.linspace(yl, yr, num = 1000)
py = normal.cdf(ynormp)
figx, ax = plt.subplots(3, 1)
ax[0].set_xlim((xl, xr))
ax[0].plot(xnormp, px, color = 'red')
ax[0].set_title('Phi and Empirical CDF of x')
ax[0].step(np.insert(x, [0, n], [xl, xr]), np.insert(lb_const, [0,0], [0,0]))
ax[1].set_xlim((yl, yr))
ax[1].plot(ynormp, py, color = 'red')
ax[1].set_title('Phi and Empirical CDF of y')
ax[1].step(np.insert(y, [0, n], [yl, yr]), np.insert(lb_const, [0, 0], [0,0]))
ax[2].plot(x, y, color = 'blue')
ax[2].set_title('Optimal phi')
ax[2].set_xlim((min(x), max(x)))
ax[2].set_ylim((min(y), max(y)))
plt.show()
