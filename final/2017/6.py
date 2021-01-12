import numpy as np
import cvxpy as cp

from disks_data import *

C = cp.Variable((n, 2))
R = cp.Variable(n)

obj = 0
for i in range(n):
	obj += np.pi * cp.square(R[i])
obj = cp.Minimize(obj)

constraints = [R >= 0]
for i in range(k):
	constraints += [C[i, 0] == Cgiven[i, 0], C[i, 1] == Cgiven[i, 1]]
	constraints += [R[i] == Rgiven[i]]

for i in range(len(Gindexes)):
	constraints += [cp.norm(C[Gindexes[i, 0]] - C[Gindexes[i, 1]]) <= (R[Gindexes[i, 0]] + R[Gindexes[i, 1]])]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.SCS)
print(prob.value)

plot_disks(C.value, R.value, Gindexes)

obj = 0
for i in range(n):
	obj += 2 * np.pi * R[i]
obj = cp.Minimize(obj)
prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.SCS)
print(prob.value)

plot_disks(C.value, R.value, Gindexes)
