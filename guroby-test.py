import numpy as np
import random
import dimod
import hybrid
import time
import gurobipy as gpy
from gurobipy import GRB

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

    # add penalty: "penalize restarting from class 0"
    for ii in range(n-1):
        aa = ii*m # x_{i,j}=x1
        bb = aa+1   # x_{i,j+1}=x2
        cc = (ii+1)*m # x_{i+1,j}=x3
        dd = cc+1 # x_{i+q,j+1}=x4

        Q[cc][cc] += 1

        Q[aa][cc] -= 0.5
        Q[cc][aa] -= 0.5

        Q[bb][cc] -= 0.5
        Q[cc][bb] -= 0.5

        Q[aa][bb] += 0.5
        Q[bb][aa] += 0.5

    return (Q, c)

def gurobi_solver(m, n, matrix, c):
    size = n * m
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
    num_max_solutions = 10
    if num_max_solutions > 1:
        qubo_model.setParam("PoolSolutions", num_max_solutions)
        qubo_model.setParam("PoolSearchMode", 2)
        # qubo_model.setParam("PoolGap", 0.1) # 10% fidelity

    # Run the Gurobi QUBO optimization
    qubo_model.optimize()

    # Print result
    if qubo_model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        if num_max_solutions == 1:
            solution = [int(qubo_vars[i].X) for i in range(size)]
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
                print(f"solution {sol_idx+1}:\n{qubo_bitstring.reshape(n,m)}")
                print("Cost of the function:", qubo_model.PoolObjVal)
    else:
        print("No solutions found")

def main():
    m = 3
    n = 8

    matrix, c = gen_Q_staircase(m, n)
    gurobi_solver(m, n, matrix, c)

if __name__ == "__main__":
    main()