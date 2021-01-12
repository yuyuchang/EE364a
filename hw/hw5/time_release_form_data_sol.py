import cvxpy as cvx
import cvxopt
import numpy as np
import pandas as pd
from time_release_form_data import *
import matplotlib.pyplot as plt

Tther = np.array(list(range(T)))

fastest = -1
Jch_vals = []
C = np.zeros((len(Tther), T))
A = {}


for i in range(len(Tther) - 1):
	a = cvx.Variable((m, K))
	c = np.zeros((1, T))
	for k in range(K):
		P_shift = np.zeros((m, 2 * T))
		P_shift[:, tau[k] - 1: tau[k] - 1 + Tp] = P
		P_shift = P_shift[:, 0:T]
		c = c + cvx.reshape(cvx.sum(cvx.multiply(cvx.reshape(a[:, k], (m, 1))\
						@ np.ones((1, T)), P_shift), axis = 0), (1, T))

	Jch = cvx.sum(cvx.norm(a[:, 1:] - a[:, :-1], "inf"))

	problem = cvx.Problem(cvx.Minimize(Jch),\
		[c <= cmax,\
		c[0, Tther[i]:] >= cmin,\
		a >= 0])

	problem.solve(solver = cvx.ECOS)
	
	if problem.status == 'optimal' and fastest == -1: fastest = i

	Jch_vals.append(problem.value)
	C[i, :] = c.value
	A[i] = a.value
	if abs(problem.value) <= 1e-6:
		break

plt.figure()
plt.plot(Tther[: len(A)], Jch_vals)
plt.xlabel('Tther')
plt.ylabel('Jch')
plt.savefig('time_release_tradeoff.png')

plt.figure()
plt.plot(list(range(T)), C[fastest, :], label = 'Tther2')
plt.plot(list(range(T)), C[7, :], label = 'Tther8')
plt.plot(list(range(T)), C[i, :], label = 'Tther26')
plt.plot(list(range(T)), cmin * np.ones(T), ':')
plt.plot(list(range(T)), cmax * np.ones(T), ':')
plt.xlabel('t')
plt.ylabel('ct')
plt.legend(['Tther2', 'Tther8', 'Tther26'])
plt.savefig('time_release_bloodstream.png')

plt.figure()
plt.subplot(3,1,1)
for j in range(6):
	plt.plot(list(range(K)), A[fastest][j], 'k')
plt.ylabel('Tther2')
plt.axis([1, K, 0, 40])
plt.subplot(3,1,2)
for j in range(6):
	plt.plot(list(range(K)), A[7][j], 'k')
plt.ylabel('Tther8')
plt.axis([1, K, 0, 40])
plt.subplot(3,1,3)
for j in range(6):
	plt.plot(list(range(K)), A[i][j], 'k')
plt.ylabel('Tther26')
plt.axis([1, K, 0, 40])
plt.xlabel('k')
plt.savefig('time_release_formulation.png')
