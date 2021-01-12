import numpy as np
import cvxpy as cp

from currency_exchange_data import *

X = cp.Variable((n, n))

obj = 0
for j in range(n):
	total = 0
	for i in range(n):
		total += (X[i, j] - X[j, i] / F[j, i])
	obj += (total * ((F[j, 0] / F[0, j]) ** 0.5))

obj = cp.Minimize(obj)

constraints = []
for i in range(n):
	for j in range(n):
		if i != j:
			constraints += [X[i, j] >= 0]

for i in range(n):
	constraints += [X[i, i] == 0]

for j in range(n):
	total = 0
	for i in range(n):
		total += (X[i, j] - X[j, i] / F[j, i])
	constraints += [c_init[j] - total >= c_req[j]]

for j in range(n):
	total = 0
	for i in range(n):
		total += X[i, j]
	constraints += [c_init[j] >= total]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.SCS)
print(prob.value)
print(X.value)
