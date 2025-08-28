import numpy as np
import math
import itertools
import pandas as pd
from scipy import stats
from sklearn import metrics
from src.select_data import *
from src.check_constraints import *
from cost_function import compute_upper_thrs, compute_lower_thrs
import time

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

def test_submatrix_penalties():
    for x in itertools.product([0, 1], repeat=4):
        a = (1-x[3-1]-x[4-1])*x[1-1]+x[3-1]*x[4-1]
        b = (1-x[1-1]-x[2-1])*x[4-1]+x[1-1]*x[2-1]
        c = x[2-1]*x[3-1]
        d = (1-x[1-1]-x[2-1])*x[3-1]+x[1-1]*x[2-1]
        print("Submatrix: ")
        print(np.array(tuple(itertools.islice(x, 4))).reshape(2, 2))
        print(f"a={a}, b={b}, c={c}, d={d}")
    return

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
    if verbose:
        print(confusion_matrix)

    return confusion_matrix

def stat_conf_matrix(n_trials):

    config = read_config()

    n = config['n_counterpart']
    grades = config['grades']

    dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    cum = conf_matrix(grades, n, default)
    for i in range(n_trials-1):
        dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
        default = dataset['default'].to_numpy().reshape(n,1)
        cum += conf_matrix(grades, n, default)
    print(cum)

def from_random_to_databse(config, default):

    # read dataset
    dataset = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['ID_Fittizio', 'DEFAULT_FLAG_rett_fact', 'score_quant_integrato'])
    dataset = dataset.rename(columns={'ID_Fittizio':'counterpart_id', 'DEFAULT_FLAG_rett_fact':'default', 'score_quant_integrato':'score'})
   
    # select optional attributes
    if config['attributes']['years']:
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['perf_year'])
        opt_attr = opt_attr.rename(columns={'perf_year':'year'})
        dataset['year'] = opt_attr['year'].values
        dataset = dataset[dataset['year'].isin(config['attributes']['years'])]

    # sort dataset by score
    dataset = dataset.sort_values(by='score')

    # Extract counterparts from dataset
    couterparts = []
    used_ids = set()
    min_score = -100
    for val in default:
        # select the first free counterpart with that default
        row = dataset[(dataset["default"] == val) & (~dataset["counterpart_id"].isin(used_ids) & (dataset["score"] > min_score))].head(1)
        if not row.empty:
            couterparts.append(row.iloc[0])
            used_ids.add(row.iloc[0]["counterpart_id"])
            min_score = row.iloc[0]["score"]

    # formatting result
    result = pd.DataFrame(couterparts)
    cols = ["counterpart_id", "year"]
    result[cols] = result[cols].astype(int)

    return result

def main():

    config = read_config()

    n = config['n_counterpart']
    grades = config['grades']

    dataset = generate_data_var_prob(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    min_thr = compute_lower_thrs(n)
    max_thr = compute_upper_thrs(n, grades)

    #--------------------------------------

    # TEST ONE RANDOM SETUP
    # generate a staircase matrix and test if the other constraints are fullfilled
    print("Testing one random setup...")
    matrix = generate_staircase_matrix(grades, n)
    # print("default:")
    # print(np.array(default).T)
    # print("matrix:")
    # print(matrix)
    test_one_solution(matrix, config, n, grades, default, max_thr, min_thr, True)

    new_df = from_random_to_databse(config, default.flatten().tolist())
    new_df.drop("year", axis=1, inplace=True)
    new_df = new_df[['counterpart_id', 'score', 'default']]
    new_df.to_csv("data/selected_counterparts.csv", index=False)
    # print(new_df)

    #--------------------------------------

    # TEST ON CONFUSION MATRIX
    # conf_matrix(grades, n, default, True)
    # stat_conf_matrix(50)

if __name__ == '__main__':
    main()
