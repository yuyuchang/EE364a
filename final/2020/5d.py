import numpy as np
import cvxpy as cp

from ranked_lists_inconsistent_data import *

obj = 0
s = cp.Variable(n)
s_matrix = cp.Variable((k, m))

for i in range(m):
  obj += cp.pos(cp.max(s_matrix[1:, i] - s_matrix[:-1, i]))
obj = cp.Minimize(obj)
constraints = []
for i in range(k):
  for j in range(m):
    constraints += [s_matrix[i, j] == s[Sigma[i, j] - 1]]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.ECOS)
print("The ordering is {}".format(np.argsort(-s.value) + 1))
print(s.value)

idx2pos = {}
for i in range(len(s.value)):
  idx2pos[np.argsort(-s.value)[i] + 1] = i

total = 0
for i in range(m):
  for j in range(k - 1):
    if idx2pos[Sigma[j + 1, i]] < idx2pos[Sigma[j, i]]:
      total += 1
      break

print("The number of inconsistent lists is: {}".format(total))
