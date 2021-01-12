import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt

from various_obj_regulator_data import *

u_a = cp.Variable((m, T))
u_b = cp.Variable((m, T))
u_c = cp.Variable((m, T))
u_d = cp.Variable((m, T))

x_a = cp.Variable((n, T))
x_b = cp.Variable((n, T))
x_c = cp.Variable((n, T))
x_d = cp.Variable((n, T))

obj_a = cp.Minimize(cp.sum(u_a ** 2))
obj_b = cp.Minimize(cp.sum(cp.norm(u_b, 2, axis = 0)))
obj_c = cp.Minimize(cp.max(cp.norm(u_c, axis = 0)))
obj_d = cp.Minimize(cp.sum(cp.norm(u_d, 1, axis = 0)))

constraints_a = [x_a[:, 0] == x_init, x_a[:, -1] == np.zeros(n)]
constraints_b = [x_b[:, 0] == x_init, x_b[:, -1] == np.zeros(n)]
constraints_c = [x_c[:, 0] == x_init, x_c[:, -1] == np.zeros(n)]
constraints_d = [x_d[:, 0] == x_init, x_d[:, -1] == np.zeros(n)]

for t in range(T - 1):
	constraints_a += [x_a[:, t + 1] == A @ x_a[:, t] + B @ u_a[:, t]]
	constraints_b += [x_b[:, t + 1] == A @ x_b[:, t] + B @ u_b[:, t]]
	constraints_c += [x_c[:, t + 1] == A @ x_c[:, t] + B @ u_c[:, t]]
	constraints_d += [x_d[:, t + 1] == A @ x_d[:, t] + B @ u_d[:, t]]

prob_a = cp.Problem(obj_a, constraints_a).solve(solver = cp.CVXOPT)
prob_b = cp.Problem(obj_b, constraints_b).solve(solver = cp.CVXOPT)
prob_c = cp.Problem(obj_c, constraints_c).solve(solver = cp.CVXOPT)
prob_d = cp.Problem(obj_d, constraints_d).solve(solver = cp.CVXOPT)

plt.figure()
plt.subplot(4,1,1)
plt.plot(np.arange(T), np.linalg.norm(u_a.value, axis = 0))
plt.subplot(4,1,2)
plt.plot(np.arange(T), np.linalg.norm(u_b.value, axis = 0))
plt.subplot(4,1,3)
plt.plot(np.arange(T), np.linalg.norm(u_c.value, axis = 0))
plt.subplot(4,1,4)
plt.plot(np.arange(T), np.linalg.norm(u_d.value, axis = 0))

plt.show()
