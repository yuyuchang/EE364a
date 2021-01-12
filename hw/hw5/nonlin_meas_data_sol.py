import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

from nonlin_meas_data import *

x = cvx.Variable(n)
z = cvx.Variable(m)

B = np.zeros((m - 1, m))
for i in range(m - 1):
  B[i, i] = -1
  B[i, i + 1] = 1

nom_cost = cvx.Problem(cvx.Minimize(cvx.norm(z - A * x)),\
        [(1 / beta) * B @ y <= B @ z,\
        B @ z <= (1 / alpha) * B @ y]).solve(solver=cvx.CVXOPT)

print(x.value)

plt.figure()
plt.plot(z.value, y)
plt.scatter((A @ x).value, y, c = 'green', s = 1)
plt.xlabel('z')
plt.ylabel('y')
plt.savefig('A6.5.png')
plt.show()
