from src.select_data import *
from src.check_constraints import *
import dimod
import hybrid
import math
import time
import itertools

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

def lower_thrs_constr(m, n, mu=1):

    min_thr = math.floor(n*0.01) if math.floor(n*0.01) != 0 else 1

    # find the number of slack variables per constraint
    N_S1 = math.floor(1+math.log2(n-min_thr))
    dim = n*m+N_S1*m

    # initialize Q and c
    Q = np.zeros([dim, dim])
    c = m * min_thr * min_thr
    offset = n*m

    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                u2 = [i1*m+j, i2*m+j]
                if u2[0]==u2[1]: # questo l'ho modificato, forse c'era un typo
                    Q[u2[0]][u2[1]] += 1
                else:
                    Q[u2[0]][u2[1]] += 0.5
                    Q[u2[1]][u2[0]] += 0.5

    for l1 in range(N_S1):
        for l2 in range(N_S1):
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
            Q[u,u] -= 2*min_thr

    index = 0
    for l in range(N_S1):
        for j in range(m):
            Q[offset+index][offset+index] += min_thr*math.pow(2,1+math.floor((l*m+j+1)/m))
            index += 1

    for i in range(n):
        for l in range(N_S1):
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

    #-------------------------------

    # Gen Q matrix
    start_time = time.perf_counter_ns()
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
        (Q_min_thr, c_min_thr) = lower_thrs_constr(m, n, mu_min_thr_constr)
        pad = Q_min_thr.shape[0] - Q.shape[0]
        Q = np.pad(Q, pad_width=((0,pad), (0, pad)), mode='constant', constant_values=0) + Q_min_thr
        c = c + c_min_thr

    # BQM generation
    bqm = from_matrix_to_bqm(Q, c)
    end_time = time.perf_counter_ns()
    print(f"Matrix size:{Q.shape}")
    print(f"Time of generation: {(end_time - start_time)/10e9} s")

    #-------------------------------

    # Solving with brute force
    start_time = time.perf_counter_ns()
    (result_bf, cost) = brute_force_solver(Q,c,Q.shape[0])
    end_time = time.perf_counter_ns()
    if config['constraints']['min_thr'] == True:
        result_bf = result_bf[:m*n]
    print(f"\nBrute Force result:\n{result_bf.reshape(n,m)}")
    print(f"Time of brute force solution: {(end_time - start_time)/10e9} s\n")

    # Solving exactly with dwave
    start_time = time.perf_counter_ns()
    e_result = exact_solver(bqm)
    df_result = e_result.lowest().to_pandas_dataframe()
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    # Print all the solutions
    result_exact_solver = df_result.iloc[:, :m*n].to_numpy()
    # print(f"All exact solutions:\n{df_result}")
    print(f"Exact solutions with dwave: {int(result_exact_solver.size/(m*n))}")
    for sol in result_exact_solver[:]:
        print(f"solution:\n{sol.reshape(n, m)}")
    print(f"Time of all exact solutions: {elapsed_time_ns/10e9} s")
    # print(f"First solution:\n{result_exact_solver[0].reshape(n, m)}")

    # Solving with annealing 
    start_time = time.perf_counter_ns()
    result = annealer_solver(Q.shape[0], bqm, shots)
    end_time = time.perf_counter_ns()
    result_ann = np.array([int(x) for x in result.samples.first.sample.values()])[:m*n]
    annealing_matrix = result_ann.reshape(n, m)
    print(f"\nAnnealing result:\n{annealing_matrix}")    
    print(f"Time of annealing solution: {(end_time - start_time)/10e9} s\n")

    print("Result validation:")
    verbose = True
    check_staircase(annealing_matrix, verbose)
    check_concentration(annealing_matrix, m, n, alpha_conc, verbose)
    # check_concentration_approx(annealing_matrix, verbose)
    check_lower_thrs(annealing_matrix, 1, verbose)

if __name__ == '__main__':
    main()