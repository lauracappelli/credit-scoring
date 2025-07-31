import numpy as np
import math
from scipy import stats

def check_staircase(matrix, verbose=False):

    # check if each counterpart is in one class
    ones_per_row = np.sum(matrix == 1, axis=1)
    if not np.all(ones_per_row == 1):
        if verbose:
            print("\tx Error: logic constraint not respected")
            print("\t\tMore or less than one class per counterpart")
        return False

    # retreive all the 1's indexes
    index_1 = np.argmax(matrix == 1, axis=1)
    # print(index_1)

    # check the first and the last counterpart
    if index_1[0] != 0:
        if verbose:
            print("\tx Error: logic constraint not respected")
            print("\t\tError in the first counterpart")
        return False
    if index_1[-1] != matrix.shape[1]-1:
        if verbose:
            print("\tx Error: logic constraint not respected")
            print("\t\tError in the last counterpart")
        return False

    # check if the matrix is a staircase matrix
    for i, el in enumerate(index_1[1:]):
        # i = inex of the vector index_1 (from 0 to m-1)
        # el = element index_1[i+1]
        # print(f"index {i+1} contains {el}")
        if el != index_1[i] and el != index_1[i]+1:
            if verbose:
                print("\tx Error: logic constraint not respected")
                print(f"\t\tError in the counterpart {i+2}")
            return False

    if verbose:
        print("\t\u2713 Logic constraint checked")
    return True

def check_concentration(matrix, m, n, alpha_conc = 0.05, verbose=False):
    J_floor = math.floor(n*n*(alpha_conc + (1-alpha_conc)/m))
    s = 0
    for i1 in range(n):
        for i2 in range(n):
            for j in range(m):
                s = s + matrix[i1,j] * matrix[i2,j]
    if s <= J_floor:
        if verbose:
            print("\t\u2713 Concentration constraint checked")
        return True
    else:
        if verbose:
            print("\tx Error: concentration constraint not respected")
        return False

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

def check_upper_thrs(matrix, max_thrs, verbose=False):

    for ii in np.sum(matrix, axis=0):
        if ii > max_thrs:
            if verbose:
                print("\tx Error: upper threshold limit constraint not respected")
            return False
    
    if verbose:
        print("\t\u2713 Upper threshold limit constraint checked")
    return True

def check_lower_thrs(matrix, min_thrs, verbose=False):
    
    for ii in np.sum(matrix, axis=0):
        if ii < min_thrs:
            if verbose:
                print("\tx Error: lower threshold limit constraint not respected")
            return False
    
    if verbose:
        print("\t\u2713 Lower threshold limit constraint checked")
    return True

def check_heterogeneity(matrix, dr, alpha_het=0.01, verbose=False):
    # print(f"default: {dr.T}")
    grad_cardinality = np.sum(matrix, axis=0) #N_j
    grad_dr = np.sum(matrix * dr, axis=0) / grad_cardinality #l_j
    binomial_var = stats.binom.var(1, grad_dr) #sigma^2_j senza dividere per n (altrimenti si sostituisce grad_cardinality a 1)
    
    cum = 0
    t_stat = np.zeros(matrix.shape[1]-1)
    p_val = np.zeros(matrix.shape[1]-1)
    for i in range(matrix.shape[1]-1):
        n1, n2 = grad_cardinality[i], grad_cardinality[i+1]

        # t-test e p-value con varianza campionaria
        # grade1 = dr[matrix[:, i] == 1]
        # grade2 = dr[matrix[:, i+1] == 1]
        # s1, s2 = np.var(grade1, ddof=1), np.var(grade2, ddof=1)  
        # t_stat[i], p_val[i] = stats.ttest_ind(grade1, grade2, equal_var=True)

        # t-test e p-value con varianza binomiale (quella chiesta da ISP)
        s1, s2 = binomial_var[i], binomial_var[i+1] # = mean*(1-mean)
        pooled_std_dev = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
        t_stat[i] = (grad_dr[i] - grad_dr[i+1]) / (pooled_std_dev * np.sqrt(1/n1 + 1/n2))
        p_val[i] = 2 * stats.t.sf(np.abs(t_stat[i]), n1 + n2 - 2)
        if p_val[i] > alpha_het:
            if verbose:
                print("\tx Error: heterogeneous constraint not respected")
                return False

    # print("t-test", t_stat)
    # print("p_val", p_val)

    if verbose:
        print("\t\u2713 Heterogeneous constraint checked")
    return True

