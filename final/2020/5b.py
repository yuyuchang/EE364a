import numpy as np
import cvxpy as cp

from ranked_lists_data import *

s = cp.Variable(n)

obj = cp.Minimize(0)
constraints = []

for i in range(Sigma.shape[1]):
  for x in range(0, Sigma.shape[0] - 1):
    constraints += [s[Sigma[x, i] - 1] >= s[Sigma[x + 1, i] - 1] + 1]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.ECOS)
print("The ordering is: {}".format(np.argsort(-s.value) + 1))
