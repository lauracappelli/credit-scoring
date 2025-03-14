import numpy as np
import random
import dimod
import hybrid
import time

def gen_random_Q(m, c):

    # Q with random elements
    matrix = np.empty([m,m])
    for i in range(m):
        for j in range(m):
            matrix[i][j] = random.randint(0,9)

    # define binary quadratic problem
    Q_dict = {(i, j): matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] != 0}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm

def gen_random_Q_opt(m, c):

    # define binary quadratic problem
    Q_dict = {(i, j): random.randint(0,9) for i in range(m) for j in range(m)}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm

def solveWithAnnealer(m, bqm, shots):

    # Set up the sampler with an initial state
    sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=shots)
    state = hybrid.core.State.from_sample({i: 0 for i in range(m)}, bqm)
 
    # Sample the problem
    new_state = sampler.run(state).result()
 
    return new_state

def main():
    m = 100
    shots = 10000

    start_time = time.perf_counter_ns()
    #bqm = gen_random_Q(m, random.randint(0,9))
    bqm = gen_random_Q_opt(m, random.randint(0,9))
    end_time = time.perf_counter_ns()

    # result = solveWithAnnealer(m, bqm, shots)
    # result_list = [int(x) for x in result.samples.first.sample.values()]
    # print(result_list)
    elapsed_time_ns = end_time - start_time
    print(f"Matrix size:{m}*{m}")
    print(f"Time of generation: {elapsed_time_ns/10e9} s")

if __name__ == "__main__":
    main()