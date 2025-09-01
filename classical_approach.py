from src.select_data import *
from src.check_constraints import *
from cost_function import compute_lower_thrs, compute_upper_thrs
import itertools
import math
import time

def main():

    # Read iperparameters from config file
    config = read_config()
    dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)
    n = len(dataset)
    grades = config['grades']
    default = dataset['default'].to_numpy().reshape(n,1)

    # compute lower and upper thresholds:
    #  - lower threshold: 1% of the counterparts
    #    (or 1 if there are less than 100 counterparts)
    #  - upper threshold: 15% of the counterparts
    #    (if there are less than 7 grades or if the 15% is less than 0, the
    #    upper threshold is set to n-grades+1. In these situations, there aren't
    #    any integer numbers such that the constraint is fulfilled)
    min_thr = compute_lower_thrs(n)
    max_thr = compute_upper_thrs(n, grades)

    # list of valid solutions
    valid_solutions = []

    print(f"\nTesting {grades ** n} combinations With the default vector:\n {default.T}")

    start_time = time.perf_counter_ns()    

    # loop over all the possible combinations of the counterparts in the grades
    for sol in itertools.product(range(grades), repeat=n):
        
        # from itertools to numpy matrix
        matrix = np.zeros([n,grades])
        for i, el in enumerate(sol):
            # set the element [i][el] to 1 if the counterpart i is in the el-th grade
            matrix[i][el] = 1

        # execute tests: run each test only if
        #  - the test is required in the config file
        #  - previous tests did not fail
        # If a test fails, set the flag variable to False to avoid running other tests on the same combination
        flag = True
        if config['test']['logic'] and flag and not check_staircase(matrix):
            flag = False
        if config['test']['conentration'] and flag and not check_concentration(matrix, config['alpha_concentration']):
            flag = False
        if config['test']['min_thr'] and flag and not check_upper_thrs(matrix, max_thr):
            flag = False
        if config['test']['max_thr'] and flag and not check_lower_thrs(matrix, min_thr):
            flag = False
        if config['test']['heterogeneity'] and flag and not check_heterogeneity(matrix, default, config['alpha_heterogeneity']):
            flag = False
        if config['test']['homogeneity'] and flag and not check_homogeneity(matrix, default, config['alpha_homogeneity']):
            flag = False
        if config['test']['monotonicity'] and flag and not check_monotonicity(matrix, default):
            flag = False    

        # If all the tests are completed, the solution is valid
        if flag:
            valid_solutions.append(sol)

    end_time = time.perf_counter_ns()
    print(f"{len(valid_solutions)} solutions found in {(end_time-start_time)/10e9} s")
    print(f"Solutions:\n{valid_solutions}\n")

if __name__ == '__main__':
    main()
