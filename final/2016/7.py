import numpy as np
import cvxpy as cp

from image_colorization_data import *

R = cp.Variable((m, n))
G = cp.Variable((m, n))
B = cp.Variable((m, n))

obj = cp.Minimize(cp.tv(R, G, B))
constraints = [0.299 * R + 0.587 * G + 0.114 * B == M,\
				R[known_ind] == R_known,\
				G[known_ind] == G_known,\
				B[known_ind] == B_known]
for i in range(m):
	for j in range(n):
		constraints += [R[i, j] >= 0, R[i, j] <= 1]
		constraints += [G[i, j] >= 0, G[i, j] <= 1]
		constraints += [B[i, j] >= 0, B[i, j] <= 1]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.SCS)

print(R.value)
print(G.value)
print(B.value)

save_img('7.png', R.value, G.value, B.value)
