import numpy as np
import math
import itertools
from scipy import stats
from sklearn import metrics
from src.select_data import *
from src.check_constraints import *
from cost_function import compute_upper_thrs, compute_lower_thrs
import time

def check_concentration_approx(matrix, verbose=False):
    ones_per_column = np.sum(matrix == 1, axis=0)
    # print(ones_per_column)

    if np.ptp(ones_per_column) <= 1:
        if verbose:
            print("\t\u2713 Concentration (approx) constraint checked")
        return True
    else:
        if verbose:
            print("\tx Error: concentration (approx) constraint not respected")
        return False

def test_submatrix_penalties():
    for x in itertools.product([0, 1], repeat=4):
        a = (1-x[3-1]-x[4-1])*x[1-1]+x[3-1]*x[4-1]
        b = (1-x[1-1]-x[2-1])*x[4-1]+x[1-1]*x[2-1]
        c = x[2-1]*x[3-1]
        d = (1-x[1-1]-x[2-1])*x[3-1]+x[1-1]*x[2-1]
        print("Submatrix: ")
        print(np.array(tuple(itertools.islice(x, 4))).reshape(2, 2))
        print(f"a={a}, b={b}, c={c}, d={d}")
    return

def test_one_random_solution(config, grades, n, default, min_thr, max_thr):
    print("Testing one random setup...")

    matrix = generate_staircase_matrix(grades, n)
    # print(np.array(default).T)
    # print(matrix)

    start_time = time.perf_counter_ns()

    if config['test']['logic']:
        check_staircase(matrix, True)
    if config['test']['conentration']:
        check_concentration(matrix, config['alpha_concentration'], True)
    if config['test']['min_thr']:
        check_upper_thrs(matrix, max_thr, True)
    if config['test']['max_thr']:
        check_lower_thrs(matrix, min_thr, True)
    if config['test']['heterogeneity']:
        check_heterogeneity(matrix, default, config['alpha_heterogeneity'], True)
    if config['test']['homogeneity']:
        check_homogeneity(matrix, default, config['alpha_homogeneity'], True)
    if config['test']['monotonicity']:
        check_monotonicity(matrix, default, True)

    end_time = time.perf_counter_ns()

    # print("Solution:\n", np.argmax(matrix, axis=1))
    print(f"Test time: {(end_time-start_time)/10e9} s")
    return

def conf_matrix(grades, n, dr, verbose=False):
    #print(f"Testing {grades ** n} combinations...")
    real = []
    summ_list = []

    for vec in itertools.product(range(grades), repeat=n):
        # build matrix
        matrix = np.zeros([n,grades])
        for i, el in enumerate(vec):
            matrix[i][el] = 1  # counterpart i in the el-th grade
       
        # matrices that fullfilled logic constraint
        if check_staircase(matrix):

            # compute "real" value
            real.append(check_monotonicity(matrix, dr))

            # compute "predicted" value
            summ = 0
            for j in range(grades-1):
                for i_1 in range(n):
                    for i_2 in range(n):
                        summ+=(dr[i_1].item()-dr[i_2].item())*matrix[i_1,j]*matrix[i_2,j+1]
            summ_list.append(summ)

    test_min = min(summ_list)
    predicted = [el == test_min for el in summ_list]

    # print(dr.T)
    # print(real)
    # print(predicted)
    confusion_matrix = metrics.confusion_matrix(real, predicted)
    if verbose:
        print(confusion_matrix)

    return confusion_matrix

def stat_conf_matrix(n_trials):

    config = read_config()

    n = config['n_counterpart']
    grades = config['grades']

    dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    cum = conf_matrix(grades, n, default)
    for i in range(n_trials-1):
        dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
        default = dataset['default'].to_numpy().reshape(n,1)
        cum += conf_matrix(grades, n, default)
    print(cum)

def main():

    config = read_config()

    n = config['n_counterpart']
    grades = config['grades']

    dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    min_thr = compute_lower_thrs(n)
    max_thr = compute_upper_thrs(n, grades)

    # generate a staircase matrix and test if the other constraints are fullfilled
    # test_one_random_solution(config, grades, n, default, min_thr, max_thr)

    # TEST ON CONFUSION MATRIX
    # conf_matrix(grades, n, default, True)
    stat_conf_matrix(50)

if __name__ == '__main__':
    main()
