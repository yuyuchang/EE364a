import numpy as np
import cvxpy as cvx

n = 4
A_tot = 10000
alpha = cvx.Constant([0.00001, 0.01, 0.01, 0.01])
M = cvx.Constant([0.1, 5, 10, 10])
A_max = cvx.Constant([40, 40, 40, 20])

a = cvx.Variable(n, pos = True)
t = cvx.Variable(1, pos = True)

obj = cvx.Minimize(t)

constraints = [cvx.prod(a) == A_tot,\
			  a <= A_max]

left = cvx.Constant(1e-10)

for i in range(n):
	tmp = alpha[i] ** 2
	for j in range(i, n):
		tmp = cvx.multiply(tmp,cvx.square(a[j]))
	left = left + tmp	

for i in range(n - 1):
	right = 1.0
	for j in range(i + 1, n):
		right = cvx.multiply(right, cvx.square(a[j]))
	right = cvx.multiply(right, cvx.square(M[i]))
	constraints += [left <= t * right]

constraints += [left <= t * M[3] * M[3]]

problem = cvx.Problem(cvx.Minimize(t), constraints)
problem.solve(gp=True, solver = cvx.ECOS)

print("The optimal gains are: {}".format(a.value))
print("The optimal dynamic range is: {}".format(t.value ** -0.5))
