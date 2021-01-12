import numpy as np
import cvxpy as cp
from multi_risk_portfolio_data import *

w = cp.Variable((n, 1))
t = cp.Variable(1)

obj = cp.Minimize(gamma * t - mu.T @ w)
constraints = [cp.sum(w) == 1]
risks = [cp.quad_form(w, Sigma) for Sigma in (Sigma_1, Sigma_2, Sigma_3, Sigma_4, Sigma_5, Sigma_6)]
constraints += [risk <= t for risk in risks]

prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.CVXOPT)

print(w.value)
print(t.value)

for risk in constraints:
  print(risk.dual_value)

for risk in risks:
  print(risk.value)
