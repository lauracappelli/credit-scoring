import numpy as np
import random
import dimod
import hybrid
import time
import gurobipy as gpy

# staircase generation as numpy matrix
def gen_Q_staircase(m,n):

    # Iniatilize Q and c
    Q = np.zeros([n*m, n*m])
    c = 0

    # add penalty: "first counterpart in first class"
    for jj in range(1, m):
        Q[jj][jj] += 1
        Q[0][jj] -= 0.5
        Q[jj][0] -= 0.5

    # add penalty: "last counterpart in the last class"
    for jj in range(m-1):
        tt = (n-1)*m+jj
        Q[tt][tt] += 1
        Q[(n*m)-1][tt] -= 0.5
        Q[tt][(n*m)-1] -= 0.5

    # add penalty: "one counterpart per class"
    for ii in range(n):
        for jj in range(m):
            tt = ii*m+jj
            Q[tt][tt] += -1
        for jj in range(m-1):
            for kk in range(jj+1,m):
                tt = ii*m+jj
                rr = ii*m+kk
                Q[tt][rr] += 1
                Q[rr][tt] += 1
        c += 1

    # add penalty: "penalize not permitted submatrix"
    for ii in range(n-1):
        for jj in range(m-1):
            aa = ii*m+jj # x_{i,j}=x1
            bb = aa+1   # x_{i,j+1}=x2
            cc = (ii+1)*m+jj # x_{i+1,j}=x3
            dd = cc+1 # x_{i+q,j+1}=x4

            # add linear terms
            Q[aa][aa] += 1
            Q[dd][dd] += 1

            # add quadratic terms
            Q[aa][bb] += 0.5
            Q[bb][aa] += 0.5

            Q[aa][cc] -= 0.5
            Q[cc][aa] -= 0.5

            Q[aa][dd] -= 1
            Q[dd][aa] -= 1

            Q[bb][cc] += 0.5
            Q[cc][bb] += 0.5

            Q[bb][dd] -= 0.5
            Q[dd][bb] -= 0.5

            Q[cc][dd] += 0.5
            Q[dd][cc] += 0.5

    return (Q, c)

def gurobi_solver(size, matrix, c):
    # model definition
    qubo_model = gpy.Model("QCS")
    qubo_vars = qubo_model.addVars(size, vtype=gpy.GRB.BINARY, name="x")

    # cost function definition
    qubo_obj = gpy.QuadExpr()
    row_idxs, col_idxs = np.nonzero(matrix)
    for jj, kk in zip(row_idxs, col_idxs):
        coeff = matrix[jj, kk]
        qubo_obj.add(coeff * qubo_vars[jj] * qubo_vars[kk])
    qubo_model.setObjective(qubo_obj, gpy.GRB.MINIMIZE)
    
    
    # Setting solver parameters
    qubo_model.setParam("OutputFlag", 1)
    qubo_model.setParam("Seed", 0)  
    # qubo_model.setParam("TimeLimit", timelimit)
    num_solutions = 2   
    if num_solutions > 1:
        qubo_model.setParam("PoolSolutions", num_solutions)
        qubo_model.setParam("PoolSearchMode", 2)  
        qubo_model.setParam("PoolGap", 0.1) # 10%
    
    # Run the Gurobi QUBO optimization
    qubo_model.optimize()

    # Get the optimizer solution
    qubo_cost = np.inf
    if qubo_model.Status in {gpy.GRB.OPTIMAL, gpy.GRB.SUBOPTIMAL}:
        status = "SUCCESS"
    else:
        status = "FAILURE"

    if status == "FAILURE":
        solutions = [np.zeros(size)]
    else:
        nfound = min(qubo_model.SolCount, num_solutions)
        qubo_costs = []

        for sol_idx in range(nfound):
            qubo_model.setParam(gpy.GRB.Param.SolutionNumber, sol_idx)
            qubo_bitstring = np.array(
                [qubo_vars[jj].getAttr("Xn") for jj in range(size)]
            )
            qubo_costs.append(
                qubo_model.PoolObjVal + c
            )
                                                    
def main():
    m = 2
    n = 3

    matrix, c = gen_Q_staircase(m, n)
    gurobi_solver(m*n, matrix, c)

if __name__ == "__main__":
    main()