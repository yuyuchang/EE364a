import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from zero_crossings_data import *

C = np.zeros((B, n))
S = np.zeros((B, n))

for j in range(B):
	for t in range(n):
		C[j, t] = np.cos(2 * np.pi * (f_min + j + 1 - 1) * (t + 1) / n)
		S[j, t] = np.sin(2 * np.pi * (f_min + j + 1 - 1) * (t + 1) / n)

A = np.vstack((C, S))

a_b = cp.Variable(2 * B)
obj = cp.norm(a_b.T @ A)
constraints = [s.T @ (a_b.T @ A).T == n]
for i in range(2048):
	constraints += [s[i] * (a_b.T @ A)[i] >= 0]

prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver = cp.SCS)

y_hat = a_b.value.T @ A
print("Recovery error: {}".format(np.linalg.norm(y - y_hat) / np.linalg.norm(y)))

plt.figure()
plt.plot(np.arange(0, n), y, label='original')
plt.plot(np.arange(0, n), y_hat, label='recovered')
plt.legend()
plt.show()
