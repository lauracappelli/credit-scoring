from src.select_data import *
from src.check_constraints import *
from cost_function import compute_lower_thrs, compute_upper_thrs
import itertools
import math
import time

def test_random_solution(config, grades, n, default, min_thr, max_thr):
    print("Testing one random setup...")

    matrix = generate_staircase_matrix(grades, n, min_thr, max_thr)
    # print(np.array(default).T)
    # print(matrix)

    start_time = time.perf_counter_ns()

    if config['test']['logic']:
        check_staircase(matrix, True)
    if config['test']['conentration']:
        check_concentration(matrix, config['m_company'], config['n_counterpart'], config['alpha_concentration'], True)
    if config['test']['min_thr']:
        check_upper_thrs(matrix, max_thr, True)
    if config['test']['max_thr']:
        check_lower_thrs(matrix, min_thr, True)
    if config['test']['heterogeneity']:
        check_heterogeneity(matrix, default, config['alpha_heterogeneity'], True)
    if config['test']['homogeneity']:
        check_homogeneity(matrix, default, config['alpha_homogeneity'], True)

    end_time = time.perf_counter_ns()

    print("Solution:\n", np.argmax(matrix, axis=1))
    print(f"Test time: {(end_time-start_time)/10e9} s")
    return

def test_all_solutions(config, grades, n, default, min_thr, max_thr):
    print(f"Testing {grades ** n} combinations...")
    valid_solutions = []

    start_time = time.perf_counter_ns()    
    # print(np.array(default).T)

    for sol in itertools.product(range(grades), repeat=n):

        # build matrix
        matrix = np.zeros([n,grades])
        for i, el in enumerate(sol):
            matrix[i][el] = 1  # counterpart i in the el-th grade
        # print(matrix)

        # execute tests
        flag = True
        verbose = False
        if config['test']['logic'] and flag and not check_staircase(matrix, verbose):
            flag = False
        if config['test']['conentration'] and flag and not check_concentration(matrix, config['m_company'], config['n_counterpart'], config['alpha_concentration'], verbose):
            flag = False
        if config['test']['min_thr'] and flag and not check_upper_thrs(matrix, max_thr, verbose):
            flag = False
        if config['test']['max_thr'] and flag and not check_lower_thrs(matrix, min_thr, verbose):
            flag = False
        if config['test']['heterogeneity'] and flag and not check_heterogeneity(matrix, default, config['alpha_heterogeneity'], verbose):
            flag = False
        if config['test']['homogeneity'] and flag and not check_homogeneity(matrix, default, config['alpha_homogeneity'], verbose):
            flag = False

        if flag:
            valid_solutions.append(sol)

    end_time = time.perf_counter_ns()
    print(f"Solutions:\n{valid_solutions}")
    print(f"Time: {(end_time-start_time)/10e9} s")

def main():

    config = read_config()

    n = config['n_counterpart']
    grades = config['m_company']

    dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    min_thr = compute_lower_thrs(n)
    max_thr = compute_upper_thrs(n, grades)

    # test_random_solution(config, grades, n, default, min_thr, max_thr)
    test_all_solutions(config, grades, n, default, min_thr, max_thr)

if __name__ == '__main__':
    main()
