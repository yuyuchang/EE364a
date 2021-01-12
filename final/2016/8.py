import numpy as np
import cvxpy as cp

p = cp.Variable(4)

obj = 0.1 * p[0] ** 2 + 0.5 * p[0] \
	  + 0.01 * p[1] ** 2 + 0.1 * p[1] \
	  + 0.02 * p[2] ** 2 \
	  + 0.05 * p[3] ** 2 + 0.2 * p[3] \
	  - cp.geo_mean(p)

prob = cp.Problem(cp.Minimize(obj))
prob.solve(solver = cp.SCS)

price = p.value
print(price)

supply = [0.2 * price[0] + 0.5, 0.02 * price[1] + 0.1, 0.04 * price[2], 0.1 * price[3] + 0.2]

print(supply)

