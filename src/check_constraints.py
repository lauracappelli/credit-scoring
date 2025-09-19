import numpy as np
import math
import time
from scipy import stats

def check_staircase(matrix, verbose=False):
    """
    Test if the input matrix fulfills the logical constraint.

    Args:
        matrix: numpy array 2D to test
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    # check if each counterpart is in one class
    ones_per_row = np.sum(matrix == 1, axis=1)
    if not np.all(ones_per_row == 1):
        if verbose:
            print("\tx Error: logical constraint not fulfilled")
            print("\t\tMore or less than one class per counterpart")
        return False

    # retreive all the 1's indexes
    counterpart_grade = np.argmax(matrix == 1, axis=1)

    # check the first and the last counterpart
    if counterpart_grade[0] != 0:
        if verbose:
            print("\tx Error: logical constraint not fulfilled")
            print("\t\tError in the first counterpart")
        return False
    if counterpart_grade[-1] != matrix.shape[1]-1:
        if verbose:
            print("\tx Error: logical constraint not fulfilled")
            print("\t\tError in the last counterpart")
        return False

    # to verify it the matrix is a staircase matrix we check if:
    #  - the counterpart i+1 has the same grade of the counterpart i
    #  - the counterpart i+1 is in the following grade wrt the counterpart i
    for i, gr in enumerate(counterpart_grade[1:]):
        # i = index of the counterpart (from 0 to n-1)
        # gr = grade of the next (i+1) counterpart
        # print(f"counterpart {i+1} belongs to grade {gr}")
        if gr != counterpart_grade[i] and gr != counterpart_grade[i]+1:
            if verbose:
                print("\tx Error: logical constraint not fulfilled")
                print(f"\t\tError in the counterpart {i+2}")
            return False

    if verbose:
        print("\t\u2713 Logical constraint fulfilled")
    return True

def check_monotonicity(matrix, default, verbose=False):
    """
    Test if the input matrix fulfills the monotonicity constraint.

    Args:
        matrix: numpy array 2D to test
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    # print(f"default: {default.T}")
    grad_cardinality = np.sum(matrix, axis=0) #N_j
    
    # Check if all the grades are not empty
    if not ((grad_cardinality == 0) == False).all():
        if verbose:
            print("\tx Error in Monotonicity constraint: at least one grade is empty")
        return False

    # Compute all the default rates and check if they are increasing or decreasing
    # (compute both to check if they are not constant)
    grad_dr = np.sum(matrix * default, axis=0) / grad_cardinality #l_j
    # print(f"DR: {grad_dr}")
    decr =  all(grad_dr[ll] >= grad_dr[ll+1] for ll in range(len(grad_dr) - 1))
    incr =  all(grad_dr[ll] <= grad_dr[ll+1] for ll in range(len(grad_dr) - 1))

    if incr == True and decr == False:
        if verbose:
            print("\t\u2713 Monotonicity constraint fulfilled")
        return True
    else:
        if verbose:
            print("\tx Error: monotonicity constraint not fulfilled")
        return False

def check_concentration(matrix, alpha_conc = 0.05, verbose=False):
    """
    Test if the input matrix fulfills the concentration constraint.

    Args:
        matrix: numpy array 2D to test
        alpha_conc: upper bound for the adjusted Herfindahl index
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    n, m = matrix.shape

    # See formulas 69 and 70 in the WP5 report
    J_floor = math.floor(n*n*(alpha_conc + (1-alpha_conc)/m))
    my_sum = 0
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                my_sum = my_sum + matrix[i1,j] * matrix[i2,j]
    if my_sum <= J_floor:
        if verbose:
            print("\t\u2713 Concentration constraint fulfilled")
        return True
    else:
        if verbose:
            print("\tx Error: concentration constraint not fulfilled")
        return False

def check_upper_thrs(matrix, max_thrs, verbose=False):
    """
    Test if the input matrix fulfills the upper threshold constraint.

    Args:
        matrix: numpy array 2D to test
        max_thrs: upper threshold admitted
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    # compute the number of counterparts for each grade and check if it is smaller than the upper threshold
    for cntp_per_grade in np.sum(matrix, axis=0):
        if cntp_per_grade > max_thrs:
            if verbose:
                print("\tx Error: upper threshold constraint not fulfilled")
            return False
    
    if verbose:
        print("\t\u2713 Upper threshold constraint fulfilled")
    return True

def check_lower_thrs(matrix, min_thrs, verbose=False):
    """
    Test if the input matrix fulfills the lower threshold constraint.

    Args:
        matrix: numpy array 2D to test
        max_thrs: lower threshold admitted
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    # compute the number of counterparts for each grade and check if it is higher than the lower threshold
    for cntp_per_grade in np.sum(matrix, axis=0):
        if cntp_per_grade < min_thrs:
            if verbose:
                print("\tx Error: lower threshold constraint not fulfilled")
            return False
    
    if verbose:
        print("\t\u2713 Lower threshold constraint fulfilled")
    return True

def check_heterogeneity(matrix, default, alpha_het=0.01, verbose=False):
    """
    Test if the input matrix fulfills the heterogeneity constraint.

    Args:
        matrix: numpy array 2D to test
        default: default of the counterparts in the matrix
        alpha_het: alpha value of the t-test
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    # print(f"default: {default.T}")
    grad_cardinality = np.sum(matrix, axis=0) #N_j
    
    # Check if the grades are not empty
    if not ((grad_cardinality == 0) == False).all():
        if verbose:
            print("\tx Error in Heterogeneity constraint: at least one grade is empty")
        return False

    # compte the default rate of each grades
    grad_dr = np.sum(matrix * default, axis=0) / grad_cardinality #l_j
    # binomial variances following the formula 14 in the WP5 report
    # (to obtain the standard binomial variance formula sobstitute 1 with grad_cardinality)
    binomial_var = stats.binom.var(1, grad_dr)
    
    # compute the t-test for each couple of grades
    t_stat = np.zeros(matrix.shape[1]-1)
    p_val = np.zeros(matrix.shape[1]-1)
    for i in range(matrix.shape[1]-1):
        n1, n2 = grad_cardinality[i], grad_cardinality[i+1]

        # t-test and p-value (sample variance)
        # grade1 = default[matrix[:, i] == 1]
        # grade2 = default[matrix[:, i+1] == 1]
        # s1, s2 = np.var(grade1, ddof=1), np.var(grade2, ddof=1)
        # if s1 == 0 and s2 == 0:
        #     if verbose:
        #         print("\tx Error: heterogeneous constraint not fulfilled")
        #     return False
        # t_stat[i], p_val[i] = stats.ttest_ind(grade1, grade2, equal_var=True)

        # t-test and p-value (binomial variance)
        s1, s2 = binomial_var[i], binomial_var[i+1] # = mean*(1-mean)
        if s1 == 0 and s2 == 0:
            if verbose:
                print("\tx Error: heterogeneous constraint not fulfilled")
                print(f"\t\t Grades {i} and {i+1} are not heterogeneous")
            return False
        pooled_std_dev = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
        t_stat[i] = (grad_dr[i] - grad_dr[i+1]) / (pooled_std_dev * np.sqrt(1/n1 + 1/n2))
        p_val[i] = 2 * stats.t.sf(np.abs(t_stat[i]), n1 + n2 - 2)
        if p_val[i] > alpha_het:
            if verbose:
                print("\tx Error: heterogeneous constraint not fulfilled")
                print(f"\t\t Grades {i} and {i+1} are not heterogeneous")
            return False

    # print("t-test", t_stat)
    # print("p_val", p_val)

    if verbose:
        print("\t\u2713 Heterogeneous constraint fulfilled")
    return True

def check_homogeneity(matrix, default, alpha_hom=0.05, verbose=False):
    """
    Test if the input matrix fulfills the homogeneity constraint.

    Args:
        matrix: numpy array 2D to test
        default: default of the counterparts in the matrix
        alpha_het: alpha value of the z-test
        verbose: enable verbose printing
    Returns:
        bool: result of the test 
    """

    # run the test for each grade j
    for j in range(matrix.shape[1]):
        # compute the default rate of each grades
        grade_dr = default[matrix[:, j] == 1]

        # Check if the grade has at least two elements (needed to build sub-populations)
        if grade_dr.size <= 2:
            if verbose:
              print("\tx Error in homogeneity constraint: at least one grade has less than 2 elements")
            return False
        
        # compute sigma^2(j)
        l_j = np.mean(grade_dr)
        sigma2 = l_j*(1-l_j)

        # extract randomly 500 couple of sub-populations
        for i in range(500):
            # select two (not empty) random subset
            mask = np.random.choice([True, False], size=grade_dr.size)
            mask[0], mask[-1] = True, False
            
            sub1, sub2 = np.array(grade_dr[mask]), np.array(grade_dr[~mask])
            n1, n2 = sub1.size, sub2.size
            mean1, mean2 = np.mean(sub1), np.mean(sub2)
            s1, s2 = mean1*(1-mean1), mean2*(1-mean2)

            if s1 == 0 and s2 == 0:
                if verbose:
                    print("\tx Error: homogeneity constraint not fulfilled")
                    print(f"\t\t Grades {i} and {i+1} are not homogeneous")
                return False
            
            # compute z-test
            z_stat = (s1 - s2) / np.sqrt(sigma2 * (1/n1 + 1/n2))
            p_val = 2 * stats.norm.sf(abs(z_stat))

            if p_val > alpha_hom:
                if verbose:
                    print("\tx Error: homogeneity constraint not fulfilled")
                    print(f"\t\t Grades {i} and {i+1} are not homogeneous")
                return False

    if verbose:
        print("\t\u2713 Homogeneity constraint fulfilled")
    return True

def test_one_solution(matrix, config, n, grades, default, max_thr, min_thr, verbose):
    """
    Given a solution of the problem, check all the constraints as requested in
    the config file

    Args:
        matrix: numpy array 2D
        config: yaml object with the config file hyperparameters
        n: number of counterparts
        grades: number of grades
        default: default of the counterparts in the matrix
        max_thr: upper threshold
        min_thr: lower threshold
        verbose: enable verbose printing
    Returns:
        bool: result of the tests  
    """

    print("Checking the constraints...")

    start_time = time.perf_counter_ns()
    flag = True
    if config['test']['logic'] and not check_staircase(matrix, verbose):
        flag = False
    if config['test']['monotonicity'] and not check_monotonicity(matrix, default, verbose):
        flag = False
    if config['test']['concentration'] and not check_concentration(matrix, config['alpha_concentration'], verbose):
        flag = False
    if config['test']['min_thr'] and not check_upper_thrs(matrix, max_thr, verbose):
        flag = False
    if config['test']['max_thr'] and not check_lower_thrs(matrix, min_thr, verbose):
        flag = False
    if config['test']['heterogeneity'] and not check_heterogeneity(matrix, default, config['alpha_heterogeneity'], verbose):
        flag = False
    if config['test']['homogeneity'] and not check_homogeneity(matrix, default, config['alpha_homogeneity'], verbose):
        flag = False
    end_time = time.perf_counter_ns()

    print(f"Test time: {(end_time-start_time)/10e9} s")
    return flag