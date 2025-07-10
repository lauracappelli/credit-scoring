from select_data import *
import dimod
import hybrid
import math
import time

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
                u[index] = [(i1)*m+j, (i2)*m+j]
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

def check_staircase(matrix):

    # check if each counterpart is in one class
    ones_per_row = np.sum(matrix == 1, axis=1)
    if not np.all(ones_per_row == 1):
        print("Error: more or less than one class per counterpart")
        return False

    # retreive all the 1's indexes
    index_1 = np.argmax(matrix == 1, axis=1)
    # print(index_1)

    # check the first and the last counterpart
    if index_1[0] != 0:
        print("Error in the first counterpart")
        return False
    if index_1[-1] != matrix.shape[1]-1:
        print("Error in the last counterpart")
        return False

    # check if the matrix is a staircase matrix
    for i, el in enumerate(index_1[1:]):
        # i = inex of the vector index_1 (from 0 to m-1)
        # el = element index_1[i+1]
        # print(f"index {i+1} contains {el}")
        if el != index_1[i] and el != index_1[i]+1:
            print(f"Error in the counterpart {i+2}")
            return False

    print("Staircase matrix constraint checked")
    return True

def check_concentration(matrix, m, n, alpha_conc = 0.05):
    J_floor = math.floor(n*n*(alpha_conc + (1-alpha_conc)/m))
    s = 0
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                s = s + matrix[i1,j] * matrix[i2,j]
    if s <= J_floor:
        print("Concentration constraint checked")
        return True
    else:
        print("Error: concentration constraint not respected")
        return False

def check_concentration_approx(matrix):
    ones_per_column = np.sum(matrix == 1, axis=0)
    # print(ones_per_column)

    if np.ptp(ones_per_column) <= 1:
        print("Concentration (approx) constraint checked")
        return True
    else:
        print("Error: concentration (approx) constraint not respected")
        return False

def from_matrix_to_bqm(matrix, c):
    
    Q_dict = {(i, j): matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1])}# if matrix[i, j] != 0}
    #print(Q_dict)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm

def solveWithAnnealer(m, bqm, shots):

    # Set up the sampler with an initial state
    sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=shots)
    state = hybrid.core.State.from_sample({i: 0 for i in range(m)}, bqm)
 
    # Sample the problem
    new_state = sampler.run(state).result()
 
    return new_state

def exactSolver(bqm):
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(bqm)

    return sampleset

def main():

    config = read_config()
    n = config['n_counterpart']
    m = config['m_company']

    alpha_conc = config['alpha_concentration']
    shots = config['shots']

    mu_one_calss_constr = config['mu_one_calss_constr']
    mu_staircase_constr = config['mu_staircase_constr']
    mu_concentration_constr = config['mu_concentration_constr']

    # Gen Q matrix
    start_time = time.perf_counter_ns()
    (Q_one_class,c_one_class) = one_class_const(m,n,mu_one_calss_constr)
    Q_staircase = staircase_constr(m,n,mu_staircase_constr)
    (Q_conc,c_conc) = concentration_constr(m, n, mu_concentration_constr)

    Q = Q_one_class + Q_staircase + Q_conc
    c = c_one_class + c_conc

    # BQM generation
    bqm = from_matrix_to_bqm(Q, c)
    end_time = time.perf_counter_ns()
    print(f"Matrix size:{m*n}*{m*n}")
    print(f"Time of generation: {(end_time - start_time)/10e9} s")

    # Solving with annealing 
    start_time = time.perf_counter_ns()  
    result = solveWithAnnealer(m*n, bqm, shots)
    end_time = time.perf_counter_ns()
    result_list = [int(x) for x in result.samples.first.sample.values()]
    annealing_matrix = np.array(result_list).reshape(n, m)
    print(f"\nAnnealing result:\n{annealing_matrix}")    
    print(f"Time of annealing solution: {(end_time - start_time)/10e9} s\n")

    check_staircase(annealing_matrix)
    check_concentration(annealing_matrix, m, n, alpha_conc)
    check_concentration_approx(annealing_matrix)

    # solving exactly
    # start_time = time.perf_counter_ns()
    # e_result = exactSolver(bqm)
    # df_result = e_result.lowest().to_pandas_dataframe()
    # end_time = time.perf_counter_ns()
    # elapsed_time_ns = end_time - start_time
    # # print(f"\nALL Exact solutions:\n{df_result}")
    # # print first result
    # matrix = df_result.iloc[:, :m*n].to_numpy()
    # # print(f"First solution:\n{matrix[0].reshape(n, m)}")
    # # Print all the solutions
    # print(f"Exact solutions: {int(matrix.size/(m*n))}")
    # for sol in matrix[:]:
    #     print(f"solution:\n{sol.reshape(n, m)}")
    # print(f"Time of all exact solutions: {elapsed_time_ns/10e9} s")


if __name__ == '__main__':
    main()