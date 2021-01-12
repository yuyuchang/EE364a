import numpy as np
import cvxpy as cp

from correlation_bounds_data import *

constraints = []
Sigma = cp.Variable((n, n), PSD = True)
t = cp.Parameter(nonneg = True)
for i in range(m):
	constraints += [cp.quad_form(A[:, i], Sigma) == sigma[i] ** 2]
for i in range(n - 1):
	for j in range(i + 1, n):
		Sij = cp.vstack([Sigma[i, i], Sigma[j, j]])
		constraints += [cp.abs(Sigma[i, j]) <= t * cp.geo_mean(Sij)]

prob = cp.Problem(cp.Minimize(0), constraints)

l, u = 0, 1
Sigma_opt = None
while u - l > 1e-3:
	t.value = (l + u) / 2
	prob.solve(solver = cp.SCS)
	if prob.status != 'infeasible' and prob.status != 'unbounded':
		u = t.value
		Sigma_opt = Sigma.value
	else:
		l = t.value

print(t.value)
print(Sigma_opt)
