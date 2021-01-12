import numpy as np
import cvxpy as cp

from satisfy_some_constraints_data import *

x = cp.Variable(n)
mu = cp.Variable()

obj = cp.Minimize(c.T @ x)
constraints = [mu >= 0, cp.sum(cp.pos(A @ x - b + mu)) <= (m - k) * mu]
prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.CVXOPT)

print("Optimal value of lambda: {}".format(1 / mu.value))
print("Obj value: {}".format(prob.value))

cnt = 0
results = A @ x.value - b
for i in range(results.shape[0]):
	if results[i] <= 1e-5:
		cnt += 1
print("The number of constraints satisfied: {}".format(cnt))

least_violated = np.argsort(A @ x.value - b)[:k]
constraints = [A[least_violated] @ x <= b[least_violated]]
prob = cp.Problem(cp.Minimize(c.T @ x), constraints)
prob.solve()
print(prob.value)
