import cvxpy as cvx

from satisfy_some_constraints_data import *

x = cvx.Variable(n)
mu = cvx.Variable()
constraints = [cvx.sum(cvx.pos(mu + A * x - b)) <= (m - k) * mu, mu >= 0]
problem = cvx.Problem(cvx.Minimize(c.T * x), constraints)
problem.solve(solver=cvx.CVXOPT)
print("Optimal value of lambda: {}".format(1 / mu.value))
print("Objective value: {}".format(problem.value))

print("Number of constraints satisfied: {}".format(np.count_nonzero(A.dot(x.value) - b <= 1e-5)))

least_violated = np.argsort(A.dot(x.value) - b)[:k]
constraints = [A[least_violated] * x <= b[least_violated]]
problem = cvx.Problem(cvx.Minimize(c.T * x), constraints)
problem.solve(solver=cvx.CVXOPT)
print("Objective value after minimizing w.r.t k constraints: {}".format(problem.value))
