from src.select_data import *
from src.check_constraints import *
from cost_function import compute_lower_thrs, compute_upper_thrs
import itertools
import math
import time

def check_solution(matrix, config, max_thr, min_thr, default, verbose):

    flag = True
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

    return flag

def main():

    config = read_config()

    n = config['n_counterpart']
    grades = config['m_company']

    dataset = generate_data(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    min_thr = compute_lower_thrs(n)
    max_thr = compute_upper_thrs(n, grades)

    all_solutions = itertools.product(range(grades), repeat=n)
    print(f"Testing {grades ** n} combinations...")
    valid_solutions = []

    start_time = time.perf_counter_ns()    
    for sol in all_solutions:
        # build matrix
        matrix = np.zeros([n,grades])
        for i, el in enumerate(sol):
            # counterpart i in the el-th grade
            matrix[i][el] = 1

        if check_solution(matrix, config, max_thr, min_thr, default, False):
            valid_solutions.append(sol)

    end_time = time.perf_counter_ns()
    print(f"Solutions:\n{valid_solutions}")
    print(f"Time: {(end_time-start_time)/10e9} s")

if __name__ == '__main__':
    main()


# genera solo una matrice e testa


# def main2():

#     config = read_config()
#     n = config['n_counterpart']
#     grades = config['m_company']
#     dataset = generate_data(config) if config['random_data'] == 'yes' else load_data(config)    
#     default = dataset['default'].to_numpy().reshape(n,1)

#     alpha_conc = config['alpha_concentration']
#     alpha_het = config['alpha_heterogeneity']
#     alpha_hom = config['alpha_homogeneity']

#     min_thr = compute_lower_thrs(n)
#     max_thrs = compute_upper_thrs(n, grades)

#     print(f"Testing {grades ** n} combinations...")
#     valid_solutions = []

#     start_time = time.perf_counter_ns()    
#     for m in itertools.product([0, 1], repeat=n*grades):
#         matrix = np.array(m).reshape((n, grades))
#         # matrix = np.array([m[i:i+3] for i in range(0, len(m), grades)])
        
#         if check_staircase(matrix) and \
#             check_concentration(matrix, grades, n) and \
#             check_upper_thrs(matrix, max_thrs) and \
#             check_lower_thrs(matrix, min_thr):# and \
#             # check_heterogeneity(matrix, default, alpha_het) and \
#             # check_homogeneity(matrix, default, alpha_hom):

#             valid_solutions.append(matrix)

