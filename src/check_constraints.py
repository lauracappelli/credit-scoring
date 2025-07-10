import numpy as np
import math

def check_staircase(matrix, verbose=False):

    # check if each counterpart is in one class
    ones_per_row = np.sum(matrix == 1, axis=1)
    if not np.all(ones_per_row == 1):
        if verbose:
            print("Error: more or less than one class per counterpart")
        return False

    # retreive all the 1's indexes
    index_1 = np.argmax(matrix == 1, axis=1)
    # print(index_1)

    # check the first and the last counterpart
    if index_1[0] != 0:
        if verbose:
            print("Error in the first counterpart")
        return False
    if index_1[-1] != matrix.shape[1]-1:
        if verbose:
            print("Error in the last counterpart")
        return False

    # check if the matrix is a staircase matrix
    for i, el in enumerate(index_1[1:]):
        # i = inex of the vector index_1 (from 0 to m-1)
        # el = element index_1[i+1]
        # print(f"index {i+1} contains {el}")
        if el != index_1[i] and el != index_1[i]+1:
            if verbose:
                print(f"Error in the counterpart {i+2}")
            return False

    if verbose:
        print("Staircase matrix constraint checked")
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
            print("Concentration constraint checked")
        return True
    else:
        if verbose:
            print("Error: concentration constraint not respected")
        return False

def check_concentration_approx(matrix, verbose=False):
    ones_per_column = np.sum(matrix == 1, axis=0)
    # print(ones_per_column)

    if np.ptp(ones_per_column) <= 1:
        if verbose:
            print("Concentration (approx) constraint checked")
        return True
    else:
        if verbose:
            print("Error: concentration (approx) constraint not respected")
        return False