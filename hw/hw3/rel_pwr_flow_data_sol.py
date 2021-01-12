import cvxpy as cvx
from rel_pwr_flow_data import *

# nominal case
p = cvx.Variable(m)
g_nom = cvx.Variable(k)
nom_cost = cvx.Problem(cvx.Minimize(c.T * g_nom),\
        [A[:k, :] * p == -g_nom,\
        A[k:, :] * p == np.array(d.T).reshape(-1,),\
        cvx.abs(p) <= np.array(Pmax.T).reshape(-1,),\
        g_nom <= np.array(Gmax).reshape(-1,),\
        g_nom >= 0]).solve(solver=cvx.CVXOPT)

# N - 1 case
P = cvx.Variable((m,m))
g_rel = cvx.Variable((k,1))
rel_cost = cvx.Problem(cvx.Minimize(c.T * g_rel),\
        [A[:k, :] * P == -g_rel * np.ones((1, m)),\
        A[k:, :] * P == d.T * np.ones((1, m)),\
        cvx.diag(P) == 0,\
        cvx.abs(P) <= Pmax.T * np.ones((1, m)),\
        g_rel <= Gmax,\
        g_rel >= 0.]).solve(solver=cvx.CVXOPT)

print("nom_cost", nom_cost)
print("rel_cost", rel_cost)
print("g_nom", g_nom.value)
print("g_rel", g_rel.value)
