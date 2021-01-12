import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

from microgrid_data import *

##############################
# Solve (a)
##############################

p_batt = cvx.Variable(N)
p_buy = cvx.Variable(N)
p_sell = cvx.Variable(N)
p_grid = p_buy - p_sell
q = cvx.Variable(N)

cost = (R_buy.T @ p_buy - R_sell.T @ p_sell) / 4

constraints = [p_ld - p_grid - p_batt - p_pv == 0,\
				-C <= p_batt,\
				p_batt <= D,\
				0 <= q,\
				q <= Q,\
				p_buy >= 0,\
				p_sell >= 0]

for i in range(95):
	constraints += [q[i + 1] == q[i] - 0.25 * p_batt[i]]

constraints += [q[0] == q[95] - 0.25 * p_batt[95]]

problem = cvx.Problem(cvx.Minimize(cost), constraints)

opt_obj = problem.solve(solver = cvx.ECOS)
print("Minimun cost: {}\n".format(opt_obj))

##############################
# Plotting the variables
##############################

p_grid_value = p_grid.value
p_batt_value = p_batt.value
q_value = q.value

# Plot p_grid
plt.figure(figsize = fig_size)
plt.plot(p_grid_value)
plt.ylabel('Power (kW)')
plt.title('p_grid (kW)', fontsize = 19)
plt.xticks(xtick_vals, xtick_labels)
plt.axvline(partial_peak_start, linestyle = '--', color = 'black')
plt.axvline(peak_start, linestyle = '--', color = 'black')
plt.axvline(peak_end, linestyle = '--', color = 'black')
plt.axvline(partial_peak_end, linestyle = '--', color = 'black')
plt.axhline(0, color = 'black')
plt.savefig('p_grid.png')

# Plot p_batt
plt.figure(figsize = fig_size)
plt.plot(p_batt_value)
plt.ylabel('Power (kW)')
plt.title('p_batt (kW)', fontsize = 19)
plt.xticks(xtick_vals, xtick_labels)
plt.axvline(partial_peak_start, linestyle = '--', color = 'black')
plt.axvline(peak_start, linestyle = '--', color = 'black')
plt.axvline(peak_end, linestyle = '--', color = 'black')
plt.axvline(partial_peak_end, linestyle = '--', color = 'black')
plt.axhline(D, linestyle = '--', color = 'black')
plt.axhline(-C, linestyle = '--', color = 'black')
plt.axhline(0, color = 'black')
plt.savefig('p_batt.png')

# Plot q
plt.figure(figsize = fig_size)
plt.plot(q_value)
plt.ylabel('Energy (kW)')
plt.title('Battery Charge (kW)', fontsize = 19)
plt.xticks(xtick_vals, xtick_labels)
plt.axvline(partial_peak_start, linestyle = '--', color = 'black')
plt.axvline(peak_start, linestyle = '--', color = 'black')
plt.axvline(peak_end, linestyle = '--', color = 'black')
plt.axvline(partial_peak_end, linestyle = '--', color = 'black')
plt.axhline(Q, color = 'black')
plt.savefig('q.png')

##############################
# part (b)
##############################
nu = constraints[0].dual_value
LMP = 4 * nu

plt.figure(figsize=(19,5))
plt.plot(R_buy, '--', label = 'Buy Price', linewidth = 2)
plt.plot(R_sell, '--', label = 'Sell Price', linewidth = 2)
plt.plot(LMP, linewidth = 2, label = 'LMP')
plt.xlabel('Time')
plt.ylabel('Price ($/kWh)')
plt.title('Locational Marginal Price', fontsize=19)
plt.legend()
plt.xticks(xtick_vals, xtick_labels )
plt.savefig('LMP.png')

##############################
# part (c)
##############################
load_cost = nu @ p_ld
batt_cost = -nu @ p_batt_value
PV_cost = -nu @ p_pv
grid_cost = nu @ p_grid_value

print("Load cost: {}".format(load_cost))
print("Battery cost: {}".format(batt_cost))
print("PV cost: {}".format(PV_cost))
print("Effective grid cost: {}".format(grid_cost))

net_cost = grid_cost - (load_cost + batt_cost + PV_cost)
print("Grid costs - other three costs: %.2f" % (net_cost))
