import numpy as np
import cvxpy as cp
from psd_cone_approx_data import *

# K = K_{1,n}
X = cp.Variable((n, n), symmetric = True)
obj = cp.Minimize(cp.trace(C @ X))
constraints = [cp.trace(A @ X) == b, cp.diag(X) >= 0]
problem = cp.Problem(obj, constraints)
problem.solve(solver = cp.SCS)
print("The optimal value of K = K_1,n is: {}".format(problem.value))

# K = K_{2,n}
X = cp.Variable((n, n), symmetric = True)
obj = cp.Minimize(cp.trace(C @ X))
constraints = [cp.trace(A @ X) == b, cp.diag(X) >= 0]
for i in range(n):
  for j in range(i + 1, n):
    constraints += [X[[i,j],:][:,[i,j]] >> 0]
problem = cp.Problem(obj, constraints)
problem.solve(solver = cp.SCS)
print("The optimal value of K = K_2,n is: {}".format(problem.value))

# K = S_+^n
X = cp.Variable((n, n), symmetric = True)
obj = cp.Minimize(cp.trace(C @ X))
constraints = [cp.trace(A @ X) == b, X >> 0]
problem = cp.Problem(obj, constraints)
problem.solve(solver = cp.SCS)
print("The optimal value of K = S_+^n is: {}".format(problem.value))

# K = S_{2,n}^*
X_list = []
for i in range(int(n * (n - 1) / 2)):
  X_list.append(cp.Variable((n, n), symmetric = True))
X = X_list[0]
for i in range(1, 10):
  X += X_list[i]
obj = cp.Minimize(cp.trace(C @ X))
constraints = [cp.trace(A @ X) == b]
for i in range(n - 1):
  for j in range(i + 1, n):
    idx = -1
    if i == 0:
      idx = i + j - 1
    elif i == 1:
      idx = i + j + 1
    else:
      idx = i + j + 2

    constraints += [X_list[idx][[i,j],:][:,[i,j]] >> 0]
    for k in range(n):
      for l in range(n):
        if (k, l) not in ((i, i), (i, j), (j, i), (j, j))	:
          constraints += [X_list[idx][k, l] == 0]
problem = cp.Problem(obj, constraints)
problem.solve(solver = cp.SCS)
print("The optimal value of K = S^*_2,n is: {}".format(problem.value))


# K = S_{1,n}^*
X = cp.Variable((n, n), symmetric = True)
obj = cp.Minimize(cp.trace(C @ X))
constraints = [cp.trace(A @ X) == b, cp.diag(X) >= 0]
for i in range(n):
  for j in range(i + 1, n):
    constraints += [X[i, j] == 0]
    constraints += [X[j, i] == 0]
problem = cp.Problem(obj, constraints)
problem.solve(solver = cp.SCS)
print("The optimal value of K = S^*_1,n is: {}".format(problem.value))
