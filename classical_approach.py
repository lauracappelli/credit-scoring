from src.select_data import *
from src.check_constraints import *
from cost_function import compute_lower_thrs, compute_upper_thrs
import itertools
import math
import time

def main():

    config = read_config()

    n = config['n_counterpart']
    grades = config['m_company']

    dataset = generate_data(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    alpha_conc = config['alpha_concentration']
    alpha_het = config['alpha_heterogeneity']

    shots = config['shots']

    min_thr = compute_lower_thrs(n)
    max_thrs = compute_upper_thrs(n, grades)

    all_solutions = itertools.product(range(grades), repeat=n)
    print(f"Testing {grades ** n} combinations...")
    valid_solutions = []

    start_time = time.perf_counter_ns()    
    for sol in all_solutions:
        # print(sol)
        # build matrix
        matrix = np.zeros([n,grades])
        for i, el in enumerate(sol):
            # counterpart i in the el-th grade
            matrix[i][el] = 1
        
        if check_staircase(matrix) and check_concentration(matrix, grades, n) and check_upper_thrs(matrix, max_thrs) and check_lower_thrs(matrix, min_thr) and check_heterogeneity(matrix, default, alpha_het, False):
            valid_solutions.append(sol)

    end_time = time.perf_counter_ns()
    print(f"Solutions:\n{valid_solutions}")
    print(f"Time: {(end_time-start_time)/10e9} s")

if __name__ == '__main__':
    main()
