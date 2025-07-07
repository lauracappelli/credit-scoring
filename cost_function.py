from select_data import *
import dimod
import hybrid
import math
import time

def one_class_const(Q, m, n, c):
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
    return (Q, c)

def first_counterpart_const(Q, m):
    # add penalty: "first counterpart in first class"
    for jj in range(1, m):
        Q[jj][jj] += 1
        Q[0][jj] -= 0.5
        Q[jj][0] -= 0.5
    return Q

def last_counterpart_const(Q, n, m):
    # add penalty: "last counterpart in the last class"
    for jj in range(m-1):
        tt = (n-1)*m+jj
        Q[tt][tt] += 1
        Q[(n*m)-1][tt] -= 0.5
        Q[tt][(n*m)-1] -= 0.5
    return Q

def staircase_constr(Q, m, n):

    Q = first_counterpart_const(Q, m)
    Q = last_counterpart_const(Q, m, n)

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

    return Q

def concentration_constr(Q, m, n, c):
    i2j1 = [] #(i_1,i_2,j)
    u2 = [] #(u_1,u_2)
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                i2j1.append([i1+1,i2+1,j+1])
    for l in i2j1:
        u_1=(l[0]-1)*m+l[2]
        u_2=(l[1]-1)*m+l[2]
        u2.append([u_1-1,u_2-1])
        
    # add initial cost function: "adjusted herfindhal index"
    c += 1/(1-m)
    gamma = m/(m-1)
    for u_item in u2:
        if u_item[0]==u_item[1]:
            Q[u_item[0]][u_item[1]] += gamma
        else:
            Q[u_item[0]][u_item[1]] += gamma/2

    return (Q, c)

def check_staircase(matrix):

    # check if each counterpart is in one class
    ones_per_row = np.sum(matrix == 1, axis=1)
    if not np.all(ones_per_row == 1):
        print("Error: more than one class per counterpart")
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

    return True

def check_concentration(matrix):
    ones_per_column = np.sum(matrix == 1, axis=0)

    print(ones_per_column)

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
    shots = config['shots']

    # Iniatilize Q and c
    Q = np.zeros([n*m, n*m])
    c = 0

    # BQM generation
    start_time = time.perf_counter_ns()
    (Q,c) = one_class_const(Q,m,n,c)
    # Q = staircase_constr(Q,m,n)
    # (Q,c) = concentration_constr(Q,m,n,c)
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
    # check_concentration(annealing_matrix)

    # solving exactly
    start_time = time.perf_counter_ns()
    e_result = exactSolver(bqm)
    df_result = e_result.lowest().to_pandas_dataframe()
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    # print(f"\nALL Exact solutions:\n{df_result}")
    # print first result
    matrix = df_result.iloc[:, :m*n].to_numpy()
    # print(f"First solution:\n{matrix[0].reshape(n, m)}")
    # Print all the solutions
    print(f"Exact solutions: {int(matrix.size/(m*n))}")
    for sol in matrix[:]:
        print(f"solution:\n{sol.reshape(n, m)}")
    print(f"Time of all exact solutions: {elapsed_time_ns/10e9} s")


if __name__ == '__main__':
    main()