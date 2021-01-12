import numpy as np
import cvxpy as cp

from matrix_equilibration_data import *

u = cp.Variable(m)
v = cp.Variable(n)

B = A ** p

obj = 0
for i in range(m):
	for j in range(n):
		obj += cp.exp(cp.log(B[i, j]) + u[i] + v[j])
obj = cp.Minimize(obj)

constraints = [cp.sum(u) == 0, cp.sum(v) == 0]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.SCS)

D = np.zeros((m ,m))
E = np.zeros((n, n))

for i in range(m):
	D[i, i] = np.exp(u.value[i] / p)

for i in range(n):
	E[i, i] = np.exp(v.value[i] / p)

results = D @ A @ E

row_norms = np.linalg.norm(results, p, 1)
col_norms = np.linalg.norm(results.T, p, 1)

print(row_norms)
print(col_norms)
