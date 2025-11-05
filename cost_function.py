from src.select_data import *
from src.check_constraints import *
import dimod
import hybrid
import math
import time
import itertools

def first_counterpart_const(m, n, mu=1):
    # penalty: "first counterpart in first class"
    Q = np.zeros([n*m, n*m])
    
    for jj in range(1, m):
        Q[jj][jj] += 1
        Q[0][jj] -= 0.5
        Q[jj][0] -= 0.5
    return mu*Q

def last_counterpart_const(m, n, mu=1):
    # penalty: "last counterpart in the last class"
    Q = np.zeros([n*m, n*m])

    for jj in range(m-1):
        tt = (n-1)*m+jj
        Q[tt][tt] += 1
        Q[(n*m)-1][tt] -= 0.5
        Q[tt][(n*m)-1] -= 0.5
    return mu*Q

def one_class_const(m, n, mu=1):
    # penalty: "one class per counterpart"
    Q = np.zeros([n*m, n*m])
    c = 0

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

def staircase_constr(m, n, mu=1):
    # penalty: "staircase matrix"
    Q = first_counterpart_const(m,n) + last_counterpart_const(m,n)

    # penalize not permitted submatrix, where a submatrix is
    # [[x1, x1], [x3, x4]]
    for ii in range(n-1):

        # penalize: [[1 0],[0 0]], [[0 0],[0 1]], [[0 1],[1 0]]
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

        # penalize restarting from class 0
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

def monotonicity_constr(m, n, default, mu=1):
    # penalty: "monotonicity"
    Q = np.zeros([n*m, n*m])
    
    num_of_default = sum(default)
    c = (m-1)*(n-num_of_default)*num_of_default

    for j in range(m-1):
        for i1 in range(n):
            for i2 in range(n):
                u_1 = (i1)*m+(j+1)-1
                u_2 = (i2)*m+(j+1)
                if default[i1]-default[i2] == -1:
                    Q[u_1,u_2] -= 0.5
                    Q[u_2,u_1] -= 0.5
                elif default[i1]-default[i2] == +1:
                    Q[u_1,u_2] += 0.5
                    Q[u_2,u_1] += 0.5

    return (mu*Q, mu*c)

def concentration_constr(m, n, mu=1):
    # penalty: "concentration"
    Q = np.zeros([n*m, n*m])

    c = -1/(1-m)
    gamma = m/((m-1)*n*n)
    
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                u1 = (i1)*m+j
                u2 = (i2)*m+j
                if u1==u2:
                    Q[u1][u2] += gamma
                else:
                    Q[u1][u2] += gamma/2
                    Q[u2][u1] += gamma/2

    return (mu*Q, mu*c)

def compute_lower_thrs(n):
    return math.floor(n*0.01) if math.floor(n*0.01) != 0 else 1

def compute_upper_thrs(n, grades):
    return math.floor(n*0.15) if grades >= 7 and math.floor(n*0.15) != 0 else (n-grades+1)
    
def threshold_constr(m, n, offset, minmax, mu=1):

    # compute the thresholds
    if minmax == 'min':
        thr = compute_lower_thrs(n)
        slack_vars = math.floor(1+math.log2(n-thr))
    elif minmax == 'max':
        thr = compute_upper_thrs(n, m)
        slack_vars = math.floor(1+math.log2(thr))
    else:
        print("Error in threshold function call")
        sys.exit(1)

    # initialize Q and c
    dim = offset+slack_vars*m
    Q = np.zeros([dim, dim])
    c = m * thr * thr

    # quadratic term (in the variable x)
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                u2 = [i1*m+j, i2*m+j]
                Q[u2[0]][u2[1]] += 0.5
                Q[u2[1]][u2[0]] += 0.5

    # quadratic term (in the slack variable)
    for l1 in range(slack_vars):
        for l2 in range(slack_vars):
            for j in range(m):
                v2 = [l1*m+j, l2*m+j]
                tmp = math.pow(2,math.floor((v2[0]+1)/m)+math.floor((v2[1]+1)/m))
                Q[offset+v2[0]][offset+v2[1]] += 0.5*tmp
                Q[offset+v2[1]][offset+v2[0]] += 0.5*tmp

    # linear term (in the x variable)
    for i in range(n):
        for j in range(m):
            u = i*m+j
            Q[u,u] -= 2*thr

    # linear term (in the slack variable)
    index = 0
    for l in range(slack_vars):
        for j in range(m):
            Q[offset+index][offset+index] += thr*math.pow(2,1+math.floor((l*m+j+1)/m))
            index += 1

    # quadratic term (in the variables x and s)
    for i in range(n):
        for l in range(slack_vars):
            for j in range(m):
                w2 = [i*m+j, l*m+j]
                tmp = math.pow(2,1+math.floor((w2[1]+1)/m))
                Q[w2[0]][offset+w2[1]] -= -0.5*tmp
                Q[offset+w2[1]][w2[0]] -= -0.5*tmp

    return (mu*Q, mu*c)

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

    return (np.array(Ymin), Cmin)

def exact_solver(bqm):
    
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(bqm)

    return sampleset

def annealer_solver(config, n, m, default, dataset, Q_size, bqm, verbose):

    # define the initial state (all elements = 0 or random elements)
    state = hybrid.core.State.from_sample({i: 0 for i in range(Q_size)}, bqm)
    # state = hybrid.core.State.from_sample({i: np.random.randint(0, 2) for i in range(Q_size)}, bqm)

    # Solve the problem with the annealer simulator
    start_time = time.perf_counter_ns()
    sampler = hybrid.SimulatedAnnealingProblemSampler(num_reads=config['reads'], num_sweeps=config['shots'])
    annealing_result = sampler.run(state).result()
    end_time = time.perf_counter_ns()
    
    # Collect results
    print("\nRESULTS OBTAINED THROUGH THE SIMULATING ANNEALER SOLVER")
    print(f"\nTime to compute the solution: {(end_time - start_time)/10e9} s\n")
    
    all_ann_bsm = annealing_result.samples.to_pandas_dataframe()
    # best_ann_bsm = np.array([int(x) for x in annealing_result.samples.first.sample.values()])[:m*n].reshape(n, m) 

    valid_sol = 0
    for i, sample in all_ann_bsm.iterrows():
        bsm = all_ann_bsm.iloc[i, :m*n].to_numpy().astype(int).reshape(n, m)
        check_constr = test_one_solution(bsm, config, n, m, default, compute_upper_thrs(n,m), compute_lower_thrs(n), True)

        if check_constr:
            dataset[f"Ann_rating_{i+1}"] = np.argmax(bsm, axis=1) + 1
            valid_sol = valid_sol+1
            grad_cardinality = np.sum(bsm, axis=0)
            num_of_default = np.sum(bsm*default, axis=0)
            stats = pd.DataFrame({
                "Grade ID": range(1, m+1),
                "Cardinality": grad_cardinality,
                "Defaults": num_of_default,
                "Default rate": num_of_default / grad_cardinality
            })

        print(f"Solution {i+1}:")
        print(f"Energy: {sample.energy}")
        print(f"The solution is correct: {check_constr}")
        if check_constr:
            print(f"Statistics:\n{stats}")
        if verbose:
            print(f"Result matrix: \n{bsm}")
        print("--------------")

    print(f"\nValid solutions found: {valid_sol}/{config['reads']}")
    # print("\nRating scale:")
    # print(dataset.to_string(index=False))

def main():

    config = read_config()
    
    # generate a random dataset or select data from the dataset
    dataset = generate_or_load_dataset(config)
    n = config['n_counterpart']
    m = config['grades']
    default = dataset['default'].to_numpy().reshape(n,1)
    
    print("\nSELECTED INSTANCE:")
    print("Number of counterparts: ", n)
    print("Number of grades: ", m)
    print(f"The number of defaults is {np.sum(default)}")
    print("Dataset:")
    print(dataset.reset_index(drop=True))

    # set input
    alpha_conc = config['alpha_concentration']
    alpha_het = config['alpha_heterogeneity']
    alpha_hom = config['alpha_homogeneity']
    shots = config['shots']
    reads = config['reads']

    mu_one_class_constr = config['mu']['one_class']
    mu_staircase_constr = config['mu']['logic']
    mu_concentration_constr = config['mu']['concentration']
    mu_min_thr_constr = config['mu']['min_thr']
    mu_max_thr_constr = config['mu']['max_thr']
    mu_monotonicity = config['mu']['monotonicity']

    #-------------------------------

    # generate Q matrix
    start_time = time.perf_counter_ns()
    Q = np.zeros([m*n, m*n])
    c = 0
    if config['constraints']['one_class'] == True:
        (Q_one_class,c_one_class) = one_class_const(m,n,mu_one_class_constr)
        Q = Q + Q_one_class
        c = c + c_one_class
    if config['constraints']['logic'] == True:
        Q = Q + staircase_constr(m,n,mu_staircase_constr)
    if config['constraints']['concentration'] == True:
        (Q_conc,c_conc) = concentration_constr(m, n, mu_concentration_constr)
        Q = Q + Q_conc
        c = c + c_conc
    if config['constraints']['monotonicity'] == True:
        (Q_monoton, c_monoton) = monotonicity_constr(m, n, default.T.squeeze(), mu_monotonicity)
        Q = Q + Q_monoton
        c = c + c_monoton
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
    end_time = time.perf_counter_ns()

    # generate the BMQ
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, c)

    print(f"\nThe QUBO problem has {Q.shape[0]} variables")
    print(f"Time spent for the generation: {(end_time - start_time)/10e9} s")

    #-------------------------------

    # Solving with brute force
    if config['solvers']['brute_force']:
        start_time = time.perf_counter_ns()
        (result_bf, cost) = brute_force_solver(Q,c,Q.shape[0])
        end_time = time.perf_counter_ns()
        result_bf = result_bf[:m*n]

        print("\nRESULTS OBTAINED THROUGH THE BRUTE FORCE SOLVER")
        print(f"\nBinary staircase matrix:")
        print(result_bf.reshape(n,m))
        print(f"\nTime of solution: {(end_time - start_time)/10e9} s\n")
        
        # Add brute force solution to the dataset
        dataset["Brute_force_rating"] = np.argmax(result_bf.reshape(n,m), axis=1) + 1
        print("Rating scale:")
        print(dataset[["counterpart_id", "default", "score", "Brute_force_rating"]])
    
    #-------------------------------
    # Solving with Gurobi
    if config['solvers']['gurobi']:
        from other_tests import gurobi_solver
        print("\nRESULTS OBTAINED THROUGH THE GUROBI SOLVER")
        gurobi_solver(config, Q, c, default)

    #-------------------------------

    # Solving exactly with dwave
    if config['solvers']['dwave_exact']:
        start_time = time.perf_counter_ns()
        e_result = exact_solver(bqm)
        end_time = time.perf_counter_ns()
        elapsed_time_ns = end_time - start_time
        result_exact_solver = e_result.lowest().to_pandas_dataframe().iloc[:, :m*n].to_numpy()
        
        # print all the solutions
        print("\nRESULTS OBTAINED THROUGH THE DWAVE EXACT SOLVER")
        print(f"\nBinary staircase matrices: {int(result_exact_solver.size/(m*n))}\n")
        for i, sol in enumerate(result_exact_solver[:]):
            print(f"solution {i+1}:\n{sol.reshape(n, m)}")

        print(f"\nTime to compute all exact solutions: {elapsed_time_ns/10e9} s")

        # Add the first solution to the dataset
        dataset["DWave_Brute_force_rating"] = np.argmax(result_exact_solver[0].reshape(n,m), axis=1) + 1
        print("Rating scale:")
        print(dataset[["counterpart_id", "default", "score", "DWave_Brute_force_rating"]])

    #-------------------------------
    # Solving with annealing
    if config['solvers']['annealing']:
        annealer_solver(config, n, m, default, dataset, Q.shape[0], bqm, False)

if __name__ == '__main__':
    main()