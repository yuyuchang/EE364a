# basic data for multi-period liability clearing problem

import numpy as np
import cvxpy as cp

n = 7
L1 = np.array(
    [[0., 90, 0, 3, 79, 0, 0],
     [57, 0, 69, 37, 0, 94, 56],
     [79, 53, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 73, 20],
     [0, 0, 42, 0, 0, 0, 90],
     [0, 34, 0, 0, 13, 0, 0],
     [38, 0, 0, 94, 85, 22, 0]]
)
c1 = np.array([10., 146., 30., 10., 10., 10., 83.])

np.testing.assert_array_less(L1 @ np.ones(n) - L1.T @ np.ones(n), c1)

for T in range(1, 100):
  P = []
  L = []
  c = []
  for i in range(T):
    P.append(cp.Variable((n, n)))
  for i in range(T + 1):
    L.append(cp.Variable((n, n,)))
  for i in range(T + 1):
    c.append(cp.Variable(n))
  constraints = [L[0] == L1, c[0] == c1]
  for t in range(T):
    constraints += [L[t + 1] == L[t] - P[t]]
    constraints += [c[t + 1] == c[t] - P[t] @ np.ones(n) + P[t].T @ np.ones(n)]
    constraints += [P[t] @ np.ones(n) <= c[t]]
  constraints += [L[-1] == 0]
  obj = cp.Minimize(0)
  prob = cp.Problem(obj, constraints)
  prob.solve(solver = cp.SCS)
  if prob.status != 'infeasible':
    print("Minimum T is: {}".format(T + 1))
    break
