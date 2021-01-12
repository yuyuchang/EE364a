import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from A9_5a import *
from A9_5b import *

def lp_solve(A, b, c):
  m, n = A.shape
  b = b.reshape(m, 1)
  nsteps = np.zeros(2)
  x0 = np.linalg.lstsq(A, b)[0]
  t0 = 2 + max(0, -min(x0))

  A1 = np.hstack((A, -np.dot(A, np.ones(n)).reshape(m, 1)))
  b1 = b - np.dot(A, np.ones(n)).reshape(m, 1)
  z0 = x0.reshape(n, 1) + t0 * np.ones((n, 1)) - np.ones((n, 1))
  c1 = np.vstack((z0, t0)).reshape(n + 1, 1)
  x_0 = np.vstack((z0, t0)).reshape(n + 1, 1)
  z_star, gap, num_newton_steps, duality_gaps =\
    lp_barrier(A1, b1, c1, x_0)
  nsteps[0] = sum(num_newton_steps)
  if len(z_star) == 0:
    print("Phase I: problem is infeasible.")
    return np.array([]), np.inf, np.inf, "Infeasible", nsteps

  print("Phase I: feasible point found.")
  x_0 = z_star[:n] - z_star[n][0] * np.ones((n, 1)) + np.ones((n, 1))

  x_star, gap, num_newton_steps, duality_gaps =\
    lp_barrier(A, b, c, x_0)
  nsteps[1] = sum(num_newton_steps)
  if len(x_star) == 0:
    return np.array([]), np.inf, np.inf, "Infeasible", nsteps

  p_star = np.dot(c.reshape(len(c)), x_star.reshape(len(c)))
  return x_star, p_star, gap, "Optimal", nsteps

if __name__ == '__main__':
  m = 100
  n = 500

  #Infeasible problem
  A = np.vstack((np.random.randn(m - 1, n), np.ones((1, n))))
  b = np.random.randn(m, 1)
  c = np.random.randn(n, 1)
  x_star, p_star, gap, status, nsteps = lp_solve(A, b, c)

  # Compare to CVXPY
  x = cp.Variable((n, 1))
  obj = cp.Minimize(c.T @ x)
  prob = cp.Problem(obj, [A * x == b, x >= 0])
  prob.solve()

  print("Status from lp solver: {}".format(status))
  print("Status from CVXPY: {}".format(prob.status))

  # Feasible problem
  A = np.vstack((np.random.randn(m - 1, n), np.ones((1, n))))
  v = np.random.rand(n) + 0.1
  b = np.dot(A, v)
  c = np.random.randn(n)
  x_star, p_star, gap, status, nsteps = lp_solve(A, b, c)


  # Compare to CVXPY
  x = cp.Variable((n, 1))
  obj = cp.Minimize(c.T @ x)
  prob = cp.Problem(obj, [A * x == b.reshape((-1, 1)), x >= 0])
  prob.solve()
  print("Optimal value from barrier method: {}"\
    .format(np.dot(c.reshape(n), x_star.reshape(n))))
  print("Optimal value from CVXPY: {}".format(prob.value))
  print("Duality gap from barrier method: {}".format(gap))
