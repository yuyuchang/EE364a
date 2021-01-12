import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

K = 500
wp = np.pi / 3
wc = .4 * np.pi
alpha = 0.0316
w = np.linspace(0, np.pi, K).reshape((-1, 1))
wi = np.max(np.where(w <= wp)[0])
wo = np.min(np.where(w >= wc)[0])

H_final = None

for N in range(1, 50):
	k = np.array(list(range(0, N + 1, 1))).reshape((-1, 1)).T
	C = np.cos(w @ k)

	a = cvx.Variable(N + 1)
	problem = cvx.Problem(cvx.Minimize(0),\
		[C[:wi, :] @ a <= 1.12,\
		C[:wi, :] @ a >= 0.89,\
		np.cos(wp * np.linspace(0, N, N + 1)) * a >= 0.89,\
		C[wo:, :] @ a <= alpha,\
		C[wo:, :] @ a >= -alpha,\
		np.cos(wc * np.linspace(0, N, N + 1)) * a <= alpha])
	problem.solve(solver=cvx.CVXOPT)
	if problem.status == 'optimal':
		print("The shortest filter length N = {}".format(N))
		H_final = a.value.reshape((1, -1)) @ np.cos(w @ k).T
		break

plt.figure()
plt.plot(w.T[0], H_final[0])
plt.plot([0, wp, wp], [0.89, 0.89, -alpha], ':')
plt.plot([wc, wc, np.pi], [1.12, alpha, alpha], ':')
plt.axis([0, np.pi, -alpha, 1.12])
plt.xlabel(r'$\omega$')
plt.ylabel(r'H($\omega$)')
plt.yticks([-alpha, 0, alpha, 0.89, 1, 1.12])
plt.savefig('A.12.d.png')
plt.show()
