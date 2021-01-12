import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

m_list = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
n = 10
error_list = []

iterations = 20

for m in m_list:
  error = 0
  for iteration in range(iterations):
    w = np.random.uniform(-1, 1, (n, m))
    w = w / np.linalg.norm(w, axis = 0)

    x = cp.Variable(n)
    y = cp.Variable(n)
    t = cp.Variable(m)

    obj = cp.Minimize(cp.sum(t) / m)
    constraints = [t >= 0,\
				  x >= 0,\
				  y >= 0,\
				  np.ones(n).T @ x <= 1, np.ones(n).T @ y <= 1]
    for i in range(m):
      constraints += [w[:, i].T @ (y - x) <= t[i]]
      constraints += [w[:, i].T @ (y - x) >= -t[i]]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver = cp.SCS)
    error += (np.linalg.norm(x.value - (np.ones(n) / n)))
  error /= iterations
  error_list.append(error)

print(error_list)

plt.figure()
plt.plot(np.array(m_list), np.array(error_list))
plt.xlabel('m')
plt.ylabel('average error')
#plt.ylim((0.028786, 0.028788))
plt.xscale('log')
plt.xticks(m_list, m_list)
plt.savefig('6.png')
plt.show()
