import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt

from deconvolution_data import *

x_a = cp.Variable(n)
x_b = cp.Variable(n)
x_c = cp.Variable(n)

obj_a = cp.Minimize(cp.norm(x_a, 1))
obj_b = cp.Minimize(cp.norm(x_b, 2))
obj_c = cp.Minimize(cp.norm(x_c, 'inf'))

constraints_a = [Y == A @ B @ x_a]
constraints_b = [Y == A @ B @ x_b]
constraints_c = [Y == A @ B @ x_c]

prob_a = cp.Problem(obj_a, constraints_a)
prob_b = cp.Problem(obj_b, constraints_b)
prob_c = cp.Problem(obj_c, constraints_c)

prob_a.solve(solver = cp.SCS)
prob_b.solve(solver = cp.SCS)
prob_c.solve(solver = cp.SCS)

z_a = B @ x_a.value
z_b = B @ x_b.value
z_c = B @ x_c.value

plt.figure()
plt.imshow(np.reshape(z_a, (d, d)).T, cmap = 'gray', interpolation = 'nearest')
plt.savefig('1a.png')

plt.figure()
plt.imshow(np.reshape(z_b, (d, d)).T, cmap = 'gray', interpolation = 'nearest')
plt.savefig('1b.png')

plt.figure()
plt.imshow(np.reshape(z_c, (d, d)).T, cmap = 'gray', interpolation = 'nearest')
plt.savefig('1c.png')

plt.figure()
plt.imshow(np.reshape(Y, (d, d)).T, cmap = 'gray', interpolation = 'nearest')
plt.savefig('1y.png')
