import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

k = 201
t = -3 + 6 * np.arange(k) / (k - 1)
y = np.exp(t)

Tpowers = np.vstack((np.ones(k), t, t ** 2)).T

a = cvx.Variable((3, 1))
b = cvx.Variable((2, 1))
gamma = cvx.Parameter(nonneg = True)

left = cvx.abs(Tpowers * a - (y.reshape((-1, 1)) * Tpowers)\
		* cvx.vstack((np.ones((1, 1)), b)))
right = gamma * Tpowers * cvx.vstack((np.ones((1, 1)), b))

problem = cvx.Problem(cvx.Minimize(0), [left <= right])

lower_bound = 0
upper_bound = np.exp(3)
tolerance = 1e-3

while upper_bound - lower_bound >= tolerance:
  gamma.value = (upper_bound + lower_bound) / 2
  problem.solve(solver = cvx.ECOS)
  if problem.status == 'optimal':
    upper_bound = gamma.value
    a_opt = a.value
    b_opt = b.value
    obj_opt = gamma.value
  else:
    lower_bound = gamma.value

y_fit = (Tpowers @ a_opt / (Tpowers @ np.vstack((np.ones((1, 1)), b_opt))))

print("a is {}".format(a_opt))
print("b is {}".format(b_opt))
print("optimal objective value is {}".format(obj_opt))

y = y.reshape((1, -1))
y_fit = y_fit.reshape((1, -1))
t = t.reshape((1, -1))

plt.figure()
plt.plot(t[0], y[0], 'b')
plt.plot(t[0], y_fit[0], 'r+')
plt.xlabel('t')
plt.ylabel('y')
plt.savefig('optimal_rational_function.png')

plt.figure()
plt.plot(t[0], y_fit[0] - y[0])
plt.xlabel('t')
plt.ylabel('error')
plt.savefig('error.png')
