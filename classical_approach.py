from src.select_data import *
from src.check_constraints import *
import itertools
import math
import time

def main():

    config = read_config()

    n = config['n_counterpart']
    grades = config['m_company']

    alpha_conc = config['alpha_concentration']
    shots = config['shots']

    mu_one_calss_constr = config['mu_one_calss_constr']
    mu_staircase_constr = config['mu_staircase_constr']
    mu_concentration_constr = config['mu_concentration_constr']

    all_solutions = itertools.product(range(grades), repeat=n)
    valid_solutions = []
    
    for sol in all_solutions:
        # build matrix
        matrix = np.zeros([n,grades])
        for i, el in enumerate(sol):
            # counterpart i in the el-th grade
            matrix[i][el] = 1
        
        if check_staircase(matrix) and check_concentration(matrix, grades, n):
            valid_solutions.append(sol)

    print(valid_solutions)

if __name__ == '__main__':
    main()
