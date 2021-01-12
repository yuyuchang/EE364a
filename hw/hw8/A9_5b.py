import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from A9_5a import *

def lp_barrier(A, b, c, x_0):
  T_0 = 1
  mu = 20
  epsilon = 1e-3
  n = len(x_0)
  t = T_0
  x = x_0
  num_newton_steps = list()
  duality_gaps = list()
  gap = float(n) / t
  while True:
    x_star, nu_star, lambda_hist = lp_acent(A, b, t * c, x)
    if len(x_star) == 0:
      return np.array([]), gap, num_newton_steps, duality_gaps
    x = x_star
    gap = float(n) / t
    num_newton_steps.append(len(lambda_hist))
    duality_gaps.append(gap)
    if gap < epsilon:
      return x_star, gap, num_newton_steps, duality_gaps
    t *= mu

if __name__ == '__main__':
  m = 10
  n = 200
  np.random.seed(2)
  A = np.vstack((np.random.randn(m - 1, n), np.ones((1, n))))
  A = np.matrix(A)
  x_0 = np.random.rand(n, 1) + 0.1
  b = A * x_0
  c = np.random.randn(n, 1)

  x_star, nu_star, lambda_hist = lp_acent(A, b, c, x_0)
#plt.semilogy(range(1, len(lambda_hist) + 1), lambda_hist)
#plt.show()

  x_star, gap, num_newton_steps, duality_gaps\
    = lp_barrier(A, b, c, x_0)

  plt.figure()
  plt.step(np.cumsum(num_newton_steps), duality_gaps, where='post')
  plt.yscale('log')
  plt.xlabel('iterations')
  plt.ylabel('gap')
  plt.savefig('A9.5b.png')

  x = cp.Variable((n, 1))
  obj = cp.Minimize(c.T @ x)
  prob = cp.Problem(obj, [A * x == b, x >= 0])
  prob.solve()
	
  print("Optimal value from barrier method: {}"\
    .format(np.dot(c.reshape(n), x_star.reshape(n))))
  print("Optimal value from CVXPY: {}".format(prob.value))
  print("Dualty gap from barrier method: {}".format(gap))
