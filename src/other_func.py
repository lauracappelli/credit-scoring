import numpy as np
import math
import itertools
from scipy import stats
from sklearn import metrics
from src.check_constraints import *

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
    #print(confusion_matrix)

    return confusion_matrix
