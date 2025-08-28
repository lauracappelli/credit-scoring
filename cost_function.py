from src.select_data import *
from src.check_constraints import *
import dimod
import hybrid
import math
import time
import itertools
import gurobipy as gpy
from gurobipy import GRB

def one_class_const(m, n, mu=1):
    Q = np.zeros([n*m, n*m])
    c = 0

    # penalty: "one counterpart per class"
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
    return (mu*Q, mu*c)

def first_counterpart_const(m, n, mu=1):
    Q = np.zeros([n*m, n*m])
    
    # penalty: "first counterpart in first class"
    for jj in range(1, m):
        Q[jj][jj] += 1
        Q[0][jj] -= 0.5
        Q[jj][0] -= 0.5
    return mu*Q

def last_counterpart_const(m, n, mu=1):
    Q = np.zeros([n*m, n*m])

    # penalty: "last counterpart in the last class"
    for jj in range(m-1):
        tt = (n-1)*m+jj
        Q[tt][tt] += 1
        Q[(n*m)-1][tt] -= 0.5
        Q[tt][(n*m)-1] -= 0.5
    return mu*Q

def staircase_constr(m, n, mu=1):
    Q = first_counterpart_const(m,n) + last_counterpart_const(m,n)

    # penalty: "penalize not permitted submatrix"
    # a submatrix is
    # [[x1, x1], [x3, x4]]
    for ii in range(n-1):
        for jj in range(m-1):
            x1 = ii*m+jj
            x2 = x1+1
            x3 = (ii+1)*m+jj
            x4 = x3+1

            # add linear terms
            Q[x1][x1] += 1
            Q[x4][x4] += 1

            # add quadratic terms
            Q[x1][x2] += 0.5
            Q[x2][x1] += 0.5

            Q[x1][x3] -= 0.5
            Q[x3][x1] -= 0.5

            Q[x1][x4] -= 1
            Q[x4][x1] -= 1

            Q[x2][x3] += 0.5
            Q[x3][x2] += 0.5

            Q[x2][x4] -= 0.5
            Q[x4][x2] -= 0.5

            Q[x3][x4] += 0.5
            Q[x4][x3] += 0.5

    # penalty: "penalize restarting from class 0"
    for ii in range(n-1):
        x1 = ii*m
        x2 = x1+1
        x3 = (ii+1)*m

        Q[x3][x3] += 1

        Q[x1][x3] -= 0.5
        Q[x3][x1] -= 0.5

        Q[x2][x3] -= 0.5
        Q[x3][x2] -= 0.5

        Q[x1][x2] += 0.5
        Q[x2][x1] += 0.5

    return mu*Q

def concentration_constr(m, n, mu=1):
    Q = np.zeros([n*m, n*m])

    u = np.zeros([n * n * m, 2], dtype=int)
    index = 0
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                u[index] = [i1*m+j, i2*m+j]
                index += 1

    # penalty: "concentration"
    c = 1/(1-m)
    gamma = m/(m-1)
    for (u1, u2) in u:
        if u1==u2:
            Q[u1][u2] += gamma
        else:
            Q[u1][u2] += gamma/2

    return (mu*Q, mu*c)

def compute_lower_thrs(n):
    return math.floor(n*0.01) if math.floor(n*0.01) != 0 else 1

def compute_upper_thrs(n, grades):
    return math.floor(n*0.15) if grades > 7 and math.floor(n*0.15) != 0 else (n-grades+1)

def threshold_constr(m, n, offset, minmax, mu=1):

    if minmax == 'min':
        thr = compute_lower_thrs(n)
        slack_vars = math.floor(1+math.log2(n-thr)) # to check
    elif minmax == 'max':
        thr = compute_upper_thrs(n, m)
        slack_vars = math.floor(1+math.log2(thr)) # to check
    else:
        print("Error in threshold function call")
        sys.exit(1)

    # initialize Q and c
    dim = offset+slack_vars*m
    Q = np.zeros([dim, dim])
    c = m * thr * thr

    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                u2 = [i1*m+j, i2*m+j]
                if u2[0]==u2[1]: # questo l'ho modificato, forse c'era un typo
                    Q[u2[0]][u2[1]] += 1
                else:
                    Q[u2[0]][u2[1]] += 0.5
                    Q[u2[1]][u2[0]] += 0.5

    for l1 in range(slack_vars):
        for l2 in range(slack_vars):
            for j in range(m):
                v2 = [l1*m+j, l2*m+j]
                tmp = math.pow(2,math.floor((v2[0]+1)/m)+math.floor((v2[1]+1)/m))
                if v2[0]==v2[1]:
                    Q[offset+v2[0]][offset+v2[1]] += tmp
                else:
                    Q[offset+v2[0]][offset+v2[1]] += 0.5*tmp
                    Q[offset+v2[1]][offset+v2[0]] += 0.5*tmp


    for i in range(n):
        for j in range(m):
            u = i*m+j
            Q[u,u] -= 2*thr

    index = 0
    for l in range(slack_vars):
        for j in range(m):
            Q[offset+index][offset+index] += thr*math.pow(2,1+math.floor((l*m+j+1)/m))
            index += 1

    for i in range(n):
        for l in range(slack_vars):
            for j in range(m):
                w2 = [i*m+j, l*m+j]
                tmp = math.pow(2,1+math.floor((w2[1]+1)/m))
                Q[w2[0]][offset+w2[1]] -= -0.5*tmp
                Q[offset+w2[1]][w2[0]] -= -0.5*tmp

    return (mu*Q, mu*c)

def from_matrix_to_bqm(matrix, c):
    
    Q_dict = {(i, j): matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1])}# if matrix[i, j] != 0}
    #print(Q_dict)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm

def annealer_solver(dim, bqm, shots):

    # Set up the sampler with an initial state
    sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=shots)
    state = hybrid.core.State.from_sample({i: 0 for i in range(dim)}, bqm)
 
    # Sample the problem
    new_state = sampler.run(state).result()
 
    return new_state

def exact_solver(bqm):
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(bqm)

    return sampleset

def brute_force_solver(Q, c, dim):

    # compute C(Y) = (Y^T)QY + (G^T)Y + c for every Y
    Ylist = list(itertools.product([0, 1], repeat=dim))
    Cmin = float('inf')

    for ii in range(len(Ylist)):
        Y = np.array(Ylist[ii])
        Cy=(Y.dot(Q).dot(Y.transpose()))+c
        if ( Cy < Cmin ):
            Cmin = Cy
            Ymin = Y.copy()

    # alternative approach:
    # for ii, item in enumerate(itertools.product([0, 1], repeat=n*m)):
    #     Y = np.array(item)
    #     Cy=np.einsum('i,ij,j->', Y, Q, Y)+c
    #     if ( Cy < Cmin ):
    #         Cmin = Cy
    #         Ymin = Y.copy()

    return (np.array(Ymin), Cmin)

def gurobi_solver(m, n, matrix, c, gurobi_n_sol, gurobi_fidelity):
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
    num_max_solutions = gurobi_n_sol
    if num_max_solutions > 1:
        qubo_model.setParam("PoolSolutions", num_max_solutions)
        qubo_model.setParam("PoolSearchMode", 2)
        qubo_model.setParam("PoolGap", gurobi_fidelity)

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

            return np.array(solution).reshape(n, m)
        
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
            
            return nfound
    else:
        print("No solutions found")
    
def main():

    config = read_config()

    dataset = generate_data(config) if config['random_data'] == 'yes' else load_data(config)
    n = len(dataset)
    m = config['grades']
    default = dataset['default'].to_numpy().reshape(n,1)

    alpha_conc = config['alpha_concentration']
    alpha_het = config['alpha_heterogeneity']
    alpha_hom = config['alpha_homogeneity']
    shots = config['shots']

    mu_one_class_constr = config['mu']['one_class']
    mu_staircase_constr = config['mu']['logic']
    mu_concentration_constr = config['mu']['concentration']
    mu_min_thr_constr = config['mu']['min_thr']
    mu_max_thr_constr = config['mu']['max_thr']

    #-------------------------------

    # Gen Q matrix
    start_time = time.perf_counter_ns()
    Q = np.zeros([m*n, m*n])
    c = 0
    if config['constraints']['one_class'] == True:
        (Q_one_class,c_one_class) = one_class_const(m,n,mu_one_class_constr)
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

    # BQM generation
    bqm = from_matrix_to_bqm(Q, c)
    end_time = time.perf_counter_ns()
    print(f"Matrix size:{Q.shape}")
    print(f"Time of generation: {(end_time - start_time)/10e9} s")

    #-------------------------------

    # # Solving with brute force
    # start_time = time.perf_counter_ns()
    # (result_bf, cost) = brute_force_solver(Q,c,Q.shape[0])
    # end_time = time.perf_counter_ns()
    # if config['constraints']['min_thr'] == True:
    #     result_bf = result_bf[:m*n]
    # print(f"\nBrute Force result:\n{result_bf.reshape(n,m)}")
    # print(f"Time of brute force solution: {(end_time - start_time)/10e9} s\n")

    #-------------------------------

    # # Solving exactly with dwave
    # start_time = time.perf_counter_ns()
    # e_result = exact_solver(bqm)
    # df_result = e_result.lowest().to_pandas_dataframe()
    # end_time = time.perf_counter_ns()
    # elapsed_time_ns = end_time - start_time
    # # Print all the solutions
    # result_exact_solver = df_result.iloc[:, :m*n].to_numpy()
    # # print(f"All exact solutions:\n{df_result}")
    # print(f"Exact solutions with dwave: {int(result_exact_solver.size/(m*n))}")
    # for sol in result_exact_solver[:]:
    #     print(f"solution:\n{sol.reshape(n, m)}")
    # print(f"Time of all exact solutions: {elapsed_time_ns/10e9} s")
    # # print(f"First solution:\n{result_exact_solver[0].reshape(n, m)}")

    #-------------------------------

    # Solving with annealing 
    start_time = time.perf_counter_ns()
    result = annealer_solver(Q.shape[0], bqm, shots)
    end_time = time.perf_counter_ns()
    result_ann = np.array([int(x) for x in result.samples.first.sample.values()])[:m*n]
    annealing_matrix = result_ann.reshape(n, m)
    print(f"\nAnnealing result:\n{annealing_matrix}")    
    print(f"Time of annealing solution: {(end_time - start_time)/10e9} s\n")

    #-------------------------------

    # Solving with Gurobi
    # gurobi_sol = gurobi_solver(m, n, Q, c, config['gurobi_n_sol'], config['gurobi_fidelity'])

    #-------------------------------
    verbose = True
    test_one_solution(annealing_matrix, config, n, m, default, compute_upper_thrs(n,m), compute_lower_thrs(n), verbose)

if __name__ == '__main__':
    main()