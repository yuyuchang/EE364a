import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt

from pv_output_data import *

c = cp.Variable(T)
s = cp.Variable(T)
r = cp.Variable(T)

obj = cp.sum(s)
for i in range(288):
	obj += (c[int(i % 288)] - c[int((i + 1) % 288)]) ** 2
obj = cp.Minimize(obj)

constraints = [s >=0, s <= c, p == c - s + r, cp.norm(r, 1) <= 4 * T]
for i in range(T - 288):
	constraints += [c[i] == c[i + 288]]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.ECOS)

print("Average value of c: {}".format(np.mean(p)))
print("Average value of s: {}".format(np.mean(s.value)))
print("Average value of c: {}".format(np.mean(c.value)))
print("Average absolute value of r: {}".format(np.mean(np.abs(r.value))))

plt.figure()
plt.subplot(4,1,1)
plt.plot(np.arange(T), c.value)
plt.title('c')

plt.subplot(4,1,2)
plt.plot(np.arange(T), s.value)
plt.title('s')

plt.subplot(4,1,3)
plt.plot(np.arange(T), r.value)
plt.title('r')

plt.subplot(4,1,4)
plt.plot(np.arange(T), p)
plt.title('p')

plt.show()
