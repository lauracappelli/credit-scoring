import numpy as np
import random
import dimod
import hybrid
import time
import gurobipy as gpy
from gurobipy import GRB
from cost_function import *

def gurobi_solver(m, n, matrix, c, g_n_sol, g_fidelity):
    size = matrix.shape[0]
    # model definition
    qubo_model = gpy.Model("QCS")
    qubo_vars = qubo_model.addVars(size, vtype=GRB.BINARY, name="x")

    # cost function definition
    qubo_expr = gpy.QuadExpr()
    row_idxs, col_idxs = np.nonzero(matrix)
    for ii, jj in zip(row_idxs, col_idxs):
        qubo_expr.add(matrix[ii, jj] * qubo_vars[ii] * qubo_vars[jj])
    qubo_expr.addConstant(c)

    # add const function to the model
    qubo_model.setObjective(qubo_expr, GRB.MINIMIZE)

    # Setting solver parameters
    qubo_model.setParam("OutputFlag", 1) # verbosity
    qubo_model.setParam("Seed", 0)  # fix seed
    # qubo_model.setParam("TimeLimit", timelimit)
    
    # Search more than 1 solution
    num_max_solutions = g_n_sol
    if num_max_solutions > 1:
        qubo_model.setParam("PoolSolutions", num_max_solutions)
        qubo_model.setParam("PoolSearchMode", 2)
        qubo_model.setParam("PoolGap", g_fidelity)

    # Run the Gurobi QUBO optimization
    qubo_model.optimize()

    # Print result
    if qubo_model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        if num_max_solutions == 1:
            solution = [int(qubo_vars[i].X) for i in range(size)]
            if len(solution) > m*n:
                solution = solution[:m*n]
            print("\nBest solution:\n", np.array(solution).reshape(n, m))
            print("Cost of the function:", qubo_model.ObjVal)
        else:
            # select all the solutions or num_max_solutions solutions
            nfound = min(qubo_model.SolCount, num_max_solutions)

            for sol_idx in range(nfound):
                qubo_model.setParam(GRB.Param.SolutionNumber, sol_idx)
                qubo_bitstring = np.array(
                    [int(qubo_vars[jj].Xn) for jj in range(size)]
                )
                if qubo_bitstring.shape[0] > m*n:
                    qubo_bitstring = qubo_bitstring[:m*n]
                print(f"solution {sol_idx+1}:\n{qubo_bitstring.reshape(n,m)}")
                print("Cost of the function:", qubo_model.PoolObjVal)
    else:
        print("No solutions found")

def main():
    config = read_config()
    n = config['n_counterpart']
    m = config['m_company']

    alpha_conc = config['alpha_concentration']
    shots = config['shots']

    mu_one_calss_constr = config['mu']['one_calss']
    mu_staircase_constr = config['mu']['logic']
    mu_concentration_constr = config['mu']['concentration']
    mu_min_thr_constr = config['mu']['min_thr']
    mu_max_thr_constr = config['mu']['max_thr']

    #-------------------------------

    # Gen Q matrix
    Q = np.zeros([m*n, m*n])
    c = 0
    if config['constraints']['one_class'] == True:
        (Q_one_class,c_one_class) = one_class_const(m,n,mu_one_calss_constr)
        Q = Q + Q_one_class
        c = c + c_one_class
    if config['constraints']['logic'] == True:
        Q = Q + staircase_constr(m,n,mu_staircase_constr)
    if config['constraints']['conentration'] == True:
        (Q_conc,c_conc) = concentration_constr(m, n, mu_concentration_constr)
        Q = Q + Q_conc
        c = c + c_conc
    if config['constraints']['min_thr'] == True:
        (Q_min_thr, c_min_thr) = threshold_constr(m, n, Q.shape[0], 'min', mu_min_thr_constr)
        pad = Q_min_thr.shape[0] - Q.shape[0]
        Q = np.pad(Q, pad_width=((0,pad), (0, pad)), mode='constant', constant_values=0) + Q_min_thr
        c = c + c_min_thr
    if config['constraints']['max_thr'] == True:
        (Q_max_thr, c_max_thr) = threshold_constr(m, n, Q.shape[0], 'max', mu_max_thr_constr)
        pad = Q_max_thr.shape[0] - Q.shape[0]
        Q = np.pad(Q, pad_width=((0,pad), (0, pad)), mode='constant', constant_values=0) + Q_max_thr
        c = c + c_max_thr

    gurobi_solver(m, n, Q, c, config['gurobi_n_sol'], config['gurobi_fidelity'])

if __name__ == "__main__":
    main()