import numpy as np
import random
import dimod
import hybrid
import time

# numpy matrix generation, then converted in bqm
def gen_random_Q(m, c):

    # Q with random elements
    matrix = np.empty([m,m])
    for i in range(m):
        for j in range(m):
            matrix[i][j] = random.randint(0,9)

    # define binary quadratic problem
    return from_matrix_to_bqm(matrix, c)

# bqm random generation
def gen_random_Q_opt(m, c):

    # define binary quadratic problem
    Q_dict = {(i, j): random.randint(0,9) for i in range(m) for j in range(m)}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm

def from_matrix_to_bqm(matrix, c):
    
    Q_dict = {(i, j): matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] != 0}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm

# staircase generation as numpy matrix, then converted in bqm
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

    # Create the BinaryQuadraticModel
    return from_matrix_to_bqm(Q, c)
    
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
        # i = indice del vettore index_1 (da 0 a m-1 compresi)
        # el = elemento index_1[i+1]
        # print(f"indice: {i+1} del vettore contiene {el}")
        if el != index_1[i] and el != index_1[i]+1:
            print(f"Error in the counterpart {i+2}")
            return False

    return True

def main():
    m = 3
    n = 7
    shots = 1000

    # BQM generation
    start_time = time.perf_counter_ns()
    # bqm = gen_random_Q(m*n, random.randint(0,9))
    # bqm = gen_random_Q_opt(m*n, random.randint(0,9))
    bqm = gen_Q_staircase(m, n)
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
    print(f"Time of annealing solution: {(end_time - start_time)/10e9} s")

    check_staircase(annealing_matrix)

    # solving exactly
    start_time = time.perf_counter_ns()
    e_result = exactSolver(bqm)
    df_result = e_result.lowest().to_pandas_dataframe()
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    print(f"\nALL Exact solutions:\n{df_result}")
    print(f"Time of solution: {elapsed_time_ns/10e9} s")
    # print first result
    matrix = df_result.iloc[:, :m*n].to_numpy()
    print(f"First solution:\n{matrix[0].reshape(n, m)}")

    # Print all the solutions
    for sol in matrix[:]:
        print(f"solution:\n{sol.reshape(n, m)}")

if __name__ == "__main__":
    main()