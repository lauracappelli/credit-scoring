from src.select_data import *
from src.check_constraints import *
from cost_function import compute_lower_thrs, compute_upper_thrs
from math import comb
import itertools
import time

def main():

    # Read iperparameters from config file
    config = read_config()
    dataset = generate_or_load_dataset(config)
    n = config['n_counterpart']
    grades = config['grades']
    default = dataset['default'].to_numpy().reshape(n,1)

    test_logic = config['test']['logic']
    test_monotonicity = config['test']['monotonicity']
    test_heterogeneity = config['test']['heterogeneity']
    test_concentration = config['test']['concentration']
    test_min_thr = config['test']['min_thr']
    test_max_thr = config['test']['max_thr']
    test_homogeneity = config['test']['homogeneity']

    print("\nSELECTED INSTANCE:")
    print("Number of counterparts: ", n)
    print("Number of grades: ", grades)
    print(f"The number of default is {int(np.sum(default))}")
    print("Dataset:")
    print(dataset.reset_index(drop=True))

    if test_heterogeneity == True or test_homogeneity == True and n < grades * 30:
        print("\nNumber of counterpars not sufficient for homogeneity and heterogeneity constraints.")
        print("These two tests won't be performed.\n")
        test_heterogeneity = False
        test_homogeneity = False
        
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

    print(f"\nTesting {comb(n-1, grades-1)} combinations...\n")

    start_time = time.perf_counter_ns()    

    # loop over all the possible partitions
    for partition in itertools.combinations(range(1,n), grades-1):
        
        # from itertools to numpy matrix
        matrix = np.zeros([n,grades])

        split_points = (0,) + partition + (n,)
        # print(split_points)
        for grade in range(grades):
            matrix[split_points[grade]:split_points[grade+1], grade] = 1
        # print(matrix)
        # print()

        # execute tests: run each test only if
        #  - the test is required in the config file
        #  - previous tests did not fail
        # If a test fails, set the flag variable to False to avoid running other tests on the same combination
        flag = True
        if test_logic and flag and not check_staircase(matrix):
            flag = False
        if test_concentration and flag and not check_concentration(matrix, config['alpha_concentration']):
            flag = False
        if test_min_thr and flag and not check_upper_thrs(matrix, max_thr):
            flag = False
        if test_max_thr and flag and not check_lower_thrs(matrix, min_thr):
            flag = False
        if test_heterogeneity and flag and not check_heterogeneity(matrix, default, config['alpha_heterogeneity']):
            flag = False
        if test_homogeneity and flag and not check_homogeneity(matrix, default, config['alpha_homogeneity']):
            flag = False
        if test_monotonicity and flag and not check_monotonicity(matrix, default):
            flag = False    

        # If all the tests are completed, the solution is valid
        if flag:
            valid_solutions.append(partition)

    end_time = time.perf_counter_ns()

    print(f"Number of solutions: {len(valid_solutions)} - Time {(end_time-start_time)/10e9} s\n")
    if len(valid_solutions) > 0:
        for i, partition in enumerate(valid_solutions):
            split_points = (0,) + partition + (n,)
            rating_scale = np.ones(n)
            for grade in range(grades):
                rating_scale[split_points[grade]:split_points[grade+1]] = int(grade+1)
            dataset["sol_" + str(i+1)] = np.array(rating_scale, dtype=int)
        print(dataset)

if __name__ == '__main__':
    main()
