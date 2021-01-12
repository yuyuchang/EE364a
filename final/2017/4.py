import numpy as np
import cvxpy as cp

from demosaic_data import *

R = cp.Variable((m, n))
G = cp.Variable((m, n))
B = cp.Variable((m, n))

obj = cp.Minimize(cp.tv(R, G, B))
constraints = [R[R_mask] == R_raw[R_mask], G[G_mask] == G_raw[G_mask], B[B_mask] == B_raw[B_mask]]
"""
for i in range(int(HEIGHT / 2)):
	for j in range(int(WIDTH / 2)):
		constraints += [R[2 * i, 2 * j] == R[2 * i + 1, 2 * j],\
						R[2 * i, 2 * j] == R[2 * i, 2 * j + 1],\
						R[2 * i, 2 * j] == R[2 * i + 1, 2 * j + 1]]
		constraints += [B[2 * i, 2 * j] == B[2 * i + 1, 2 * j + 1],\
						B[2 * i + 1, 2 * j] == B[2 * i + 1, 2 * j + 1],\
						B[2 * i, 2 * j + 1] == B[2 * i + 1, 2 * j + 1]]
		constraints += [G[2 * i, 2 * j] == (G[2 * i + 1, 2 * j] + G[2 * i, 2 * j + 1]) / 2]
		constraints += [G[2 * i + 1, 2 * j + 1] == (G[2 * i + 1, 2 * j] + G[2 * i, 2 * j + 1]) / 2]
"""
prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.SCS)

print(prob.value)
save_image(R.value, G.value, B.value)
