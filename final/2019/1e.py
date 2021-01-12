import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from scipy.special import factorial

N = np.array([0, 4, 2, 2, 3, 0, 4, 5, 6, 6, 4, 1, 4, 4, 0, 1, 3, 4, 2, 0, 3, 2, 0, 1])
N_test = np.array([0, 1, 3, 2, 3, 1, 4, 5, 3, 1, 4, 3, 5, 5, 2, 1, 1, 1, 2, 0, 1, 2, 1, 0])

rho = [0.1, 1, 10, 100]
Lambda = cp.Variable(24)

loss = 0
regularization = 0

for i in range(24):
	loss += (Lambda[i] - N[i] * cp.log(Lambda[i]))

for i in range(24 - 1):
	regularization += (Lambda[i + 1] - Lambda[i]) ** 2
regularization += (Lambda[0] - Lambda[-1]) ** 2
constraints = [Lambda >= 0]

for r in rho:
	obj = cp.Minimize(loss + r * regularization)
	prob = cp.Problem(obj, constraints)
	prob.solve(solver = cp.ECOS)
	total = 0
	for i in range(24):
		total += (Lambda.value[i] - N_test[i] * np.log(Lambda.value[i]) + np.log(factorial(N_test[i])))
	print("The log likelihood of rho = {} is {}".format(str(r), -total))
