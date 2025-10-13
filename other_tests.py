import numpy as np
import math
import itertools
import pandas as pd
from scipy import stats
from sklearn import metrics
from src.select_data import *
from src.check_constraints import *
from cost_function import compute_upper_thrs, compute_lower_thrs, monotonicity_constr_appr
import time
import matplotlib.pyplot as plt

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

def plotting_costs(n, m, dr, mu_monoton, bsm, costs, exact, approx):
   
    # # Plot 1
    # mask = np.array([exact[i]==approx[i] for i in range(len(exact))])
    # correct_guess = costs[mask]
    # wrong_guess = costs[~mask]

    # plt.figure(figsize=(8,5))
    # plt.plot(correct_guess, marker='x', linestyle='', label='Correct guess')
    # plt.plot(wrong_guess, marker='o', linestyle='', label='Wrong guess')
    # plt.suptitle("Costs of BSMs", fontsize=10)
    # plt.title(f"n={n}, m={m}, mu={mu_monoton}, d_vec={dr}", fontsize=10)
    # plt.xlabel("Index of BSMs", fontsize=10)
    # plt.xticks(range(0,max(len(correct_guess), len(wrong_guess))))
    # plt.ylabel("Energy", fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.savefig(f"output/plots/fig-m={n}-m={m}-d_vec={dr}.png")

    # Plot 2
    tp_costs = costs[np.array([exact[i] and approx[i] for i in range(len(exact))])]
    tn_costs = costs[np.array([(not exact[i]) and (not approx[i]) for i in range(len(exact))])]
    fp_costs = costs[np.array([(not exact[i]) and approx[i] for i in range(len(exact))])]
    fn_costs = costs[np.array([exact[i] and (not approx[i]) for i in range(len(exact))])]

    data = [tp_costs, tn_costs, fp_costs, fn_costs]
    bins = np.arange(min(costs), max(costs) + 2) - 0.5
    colors = ['palegreen', 'forestgreen', 'red', 'orange']
    labels = ['tp', 'tn', 'fp', 'fn']

    plt.figure()
    plt.hist(data, stacked=True, bins=bins, color=colors, label=labels, edgecolor='black')
    plt.title(f"Costs of BSMs\nn={n}, m={m}, mu={mu_monoton}, d_vec={dr}", fontsize=10)
    plt.xlabel("Energy", fontsize=10)
    plt.xticks(np.arange(min(costs), max(costs)+1))
    plt.ylabel("Count", fontsize=10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"output/plots/hist-m={n}-m={m}-d_vec={dr}.png")

def conf_matrix(m, n, dr, verbose=False):
    if verbose:
        print(f"Testing {m ** n} combinations...")

    # instantiate lists
    bsm = []
    exact = []
    costs = []

    # build QUBO matrix with only monotonicity constraint
    mu_monoton = 1
    (Q,c) = monotonicity_constr_appr(m, n, dr, mu_monoton)

    # iteration on all the possible combinations
    for vec in itertools.product(range(m), repeat=n):
        # build matrix
        matrix = np.zeros([n,m])
        for i, el in enumerate(vec):
            matrix[i][el] = 1  # counterpart i in the el-th grade
       
        # matrices that fullfilled logic constraint
        if check_staircase(matrix):
            # add staircase matrix in the vector
            bsm.append(matrix)

            # compute "exact" value
            exact.append(check_monotonicity(matrix, dr))

            # compute the cost of the BS matrix
            x = matrix.reshape(1,m*n).squeeze()
            costs.append(int((x.dot(Q).dot(x.transpose()))+c))

    min_cost = min(costs)
    approx = [el == min_cost for el in costs]

    confusion_matrix = metrics.confusion_matrix(exact, approx)
    tn, fp, fn, tp = confusion_matrix.ravel().tolist()

    print("DEFAULT")
    print(np.array(dr.T.squeeze()))
    print("Best solution:")
    print(bsm[costs.index(min(costs))])
    print("CONFUSION MATRIX")
    print(confusion_matrix)
    print("tn, fp, fn, tp = ", tn, fp, fn, tp)

    if verbose:
        print()
        for i, el in enumerate(bsm):
            print("MATRIX ", i+1)
            print(el)
            print("Exact monoton fulfilled: ", exact[i])
            print("Approx monoton fulfilled: ", approx[i])
            print("Cost (approx method): ", costs[i])

    plotting_costs(n, m, np.array(dr.T.squeeze()), mu_monoton, bsm, np.array(costs), np.array(exact), np.array(approx))

    return (tn, fp, fn, tp)

def stat_conf_matrix(n_trials):

    config = read_config()

    n = config['n_counterpart']
    grades = config['grades']

    dataset = generate_or_load_dataset_incr_def_prob(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    cum = conf_matrix(grades, n, default)
    for i in range(n_trials-1):
        dataset = generate_or_load_dataset_incr_def_prob(config) if config['random_data'] == 'yes' else load_data(config)    
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

    dataset = generate_or_load_dataset(config)    
    # dataset = generate_or_load_dataset(config) if config['random_data'] == 'yes' else load_data(config)    
    default = dataset['default'].to_numpy().reshape(n,1)

    min_thr = compute_lower_thrs(n)
    max_thr = compute_upper_thrs(n, grades)

    #--------------------------------------

    # # TEST ONE RANDOM SETUP
    # # generate a staircase matrix and test if the other constraints are fullfilled
    # print("Testing one random setup...")
    # matrix = generate_staircase_matrix(grades, n)
    # # print("default:")
    # # print(np.array(default).T)
    # # print("matrix:")
    # # print(matrix)
    # test_one_solution(matrix, config, n, grades, default, max_thr, min_thr, True)

    # # connect to the database
    # new_df = from_random_to_databse(config, default.flatten().tolist())
    # new_df.drop("year", axis=1, inplace=True)
    # new_df = new_df[['counterpart_id', 'score', 'default']]
    # new_df.to_csv("data/selected_counterparts.csv", index=False)
    # # print(new_df)

    #--------------------------------------

    # TEST ON CONFUSION MATRIX
    conf_matrix(grades, n, default, False)
    # stat_conf_matrix(50)

if __name__ == '__main__':
    main()
