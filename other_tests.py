from src.select_data import *
from src.check_constraints import *
from cost_function import *
import numpy as np
import itertools
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import gurobipy as gpy
from gurobipy import GRB
from gurobi_optimods.qubo import solve_qubo
import optuna

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

def gurobiop_solver(matrix, m, n):
    return solve_qubo(matrix, time_limit=180)

def gurobipy_solver(config, matrix, c, default):

    m = config['grades']
    n = config['n_counterpart']
    gurobi_n_sol = config['gurobi_n_sol']
    gurobi_fidelity = config['gurobi_fidelity']

    size = matrix.shape[0]
    qubo_model = gpy.Model("QCS")
    qubo_vars = qubo_model.addVars(size, vtype=GRB.BINARY, name="x")
    sol_set = []

    # cost function definition
    qubo_expr = gpy.QuadExpr()
    row_idxs, col_idxs = np.nonzero(matrix)
    for ii, jj in zip(row_idxs, col_idxs):
        qubo_expr.add(matrix[ii, jj] * qubo_vars[ii] * qubo_vars[jj])
    qubo_expr.addConstant(c)

    # add const function to the model
    qubo_model.setObjective(qubo_expr, GRB.MINIMIZE)

    # Setting solver parameters
    qubo_model.setParam("OutputFlag", 1) # verbosity
    qubo_model.setParam("Seed", 0)  # fix seed
    qubo_model.setParam("TimeLimit", 300.0)
    # qubo_model.setParam("TimeLimit", timelimit)
    
    # Search more than 1 solution
    num_max_solutions = gurobi_n_sol
    if num_max_solutions > 1:
        qubo_model.setParam("PoolSolutions", num_max_solutions)
        qubo_model.setParam("PoolSearchMode", 2)
        qubo_model.setParam("PoolGap", gurobi_fidelity)

    # Run the Gurobi QUBO optimization
    qubo_model.optimize()

    # Print result
    if qubo_model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        if num_max_solutions == 1:
            solution = [int(qubo_vars[i].X) for i in range(size)]
            if len(solution) > m*n:
                solution = solution[:m*n]
            np_sol = np.array(solution).reshape(n, m)
            sol_set.append(np_sol)
            print("\nBest solution:\n", np_sol)
            print("Cost of the function:", qubo_model.ObjVal)
        
        else:
            # select all the solutions or num_max_solutions solutions
            nfound = min(qubo_model.SolCount, num_max_solutions)
            for sol_idx in range(nfound):
                qubo_model.setParam(GRB.Param.SolutionNumber, sol_idx)
                qubo_bitstring = np.array(
                    [int(qubo_vars[jj].Xn) for jj in range(size)]
                )
                if qubo_bitstring.shape[0] > m*n:
                    qubo_bitstring = qubo_bitstring[:m*n]
                np_sol = qubo_bitstring.reshape(n,m)
                sol_set.append(np_sol)
                print(f"solution {sol_idx+1}:\n{np_sol}")
                print("Cost of the function:", qubo_model.PoolObjVal)
        
        print("\nValidation of the gurobi result:")
        for np_sol_item in sol_set:
            is_valid = test_one_solution(np_sol_item, config, n, m, default, compute_upper_thrs(n,m), compute_lower_thrs(n), True)
    else:
        print("No solutions found")

def plotting_costs(n, m, dr, mu_monoton, costs, exact, approx):
   
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
    plt.savefig(f"output/plots/hist-n{n}-m{m}-dvec{''.join(map(str, dr))}.png")

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

    plotting_costs(n, m, np.array(dr.T.squeeze()), mu_monoton, np.array(costs), np.array(exact), np.array(approx))

    return (tn, fp, fn, tp)

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

def find_sol_annealing(config, default, n, m, mu_one_class, mu_sc_first_last_class, mu_sc_subm_1000, mu_sc_subm_0001, mu_sc_subm_0110, mu_sc_restart, mu_sc_column_one, mu_sc_change_class, mu_mon, mu_conc, mu_thr):

    Q = np.zeros([m*n, m*n])
    c = 0
    if config['constraints']['one_class'] == True:
        (Q_one_class,c_one_class) = one_class_const(m,n,mu_one_class)
        Q = Q + Q_one_class
        c = c + c_one_class
    if config['constraints']['logic'] == True:
        Q = Q + staircase_constr(m,n,mu_sc_first_last_class,mu_sc_subm_1000,mu_sc_subm_0001,mu_sc_subm_0110,mu_sc_restart,mu_sc_column_one,mu_sc_change_class)
    if config['constraints']['concentration'] == True:
        (Q_conc,c_conc) = concentration_constr(m, n, mu_conc)
        Q = Q + Q_conc
        c = c + c_conc
    if config['constraints']['monotonicity'] == True:
        (Q_monoton, c_monoton) = monotonicity_constr(m, n, default.T.squeeze(), mu_mon)
        Q = Q + Q_monoton
        c = c + c_monoton
    if config['constraints']['min_thr'] == True:
        (Q_min_thr, c_min_thr) = threshold_constr(m, n, Q.shape[0], 'min', mu_thr)
        pad = Q_min_thr.shape[0] - Q.shape[0]
        Q = np.pad(Q, pad_width=((0,pad), (0, pad)), mode='constant', constant_values=0) + Q_min_thr
        c = c + c_min_thr
    if config['constraints']['max_thr'] == True:
        (Q_max_thr, c_max_thr) = threshold_constr(m, n, Q.shape[0], 'max', mu_thr)
        pad = Q_max_thr.shape[0] - Q.shape[0]
        Q = np.pad(Q, pad_width=((0,pad), (0, pad)), mode='constant', constant_values=0) + Q_max_thr
        c = c + c_max_thr

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, c)

    state = hybrid.core.State.from_sample({i: 0 for i in range(Q.shape[0])}, bqm)
    sampler = hybrid.SimulatedAnnealingProblemSampler(num_reads=config['reads'], num_sweeps=config['shots'])
    annealing_result = sampler.run(state).result()
    # best_ann_bsm = np.array([int(x) for x in annealing_result.samples.first.sample.values()])[:m*n].reshape(n, m) 
    best_ann_bsm = annealing_result.samples.first.energy
    return best_ann_bsm

def find_mu_optuna(config, n, m, default):

    def objective(trial):
        mu_one_class = trial.suggest_int("mu_one_class", 80,120)
        mu_sc_first_last_class = trial.suggest_int("mu_sc_first_last_class", 50,60) 
        mu_sc_subm_1000 = trial.suggest_int("mu_sc_subm_1000", 50,60)
        mu_sc_subm_0001 = trial.suggest_int("mu_sc_subm_0001", 80,120)
        mu_sc_subm_0110 = trial.suggest_int("mu_sc_subm_0110", 50,60)
        mu_sc_restart = trial.suggest_int("mu_sc_restart", 100,130)
        mu_sc_column_one = trial.suggest_int("mu_sc_column_one", 100,130)
        mu_sc_change_class = trial.suggest_int("mu_sc_change_class", 100,130)
        mu_mon = trial.suggest_int("mu_mon", 5,15)
        mu_conc = trial.suggest_int("mu_conc", 5,15)
        mu_thr = trial.suggest_int("mu_thr", 5,15)

        return find_sol_annealing(config, default, n, m, mu_one_class, mu_sc_first_last_class, mu_sc_subm_1000, mu_sc_subm_0001, mu_sc_subm_0110, mu_sc_restart, mu_sc_column_one, mu_sc_change_class, mu_mon, mu_conc, mu_thr)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    return study

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
    # conf_matrix(grades, n, default, False)

    #--------------------------------------

    # OPTUNA
    study = find_mu_optuna(config, n, grades, default)
    best_value = study.best_value
    print(study.best_params, best_value)

    best_trials = [t for t in study.trials if abs(t.value - best_value) < 1e-9]

    print(f"\nNumero di configurazioni equivalenti: {len(best_trials)}\n")

    for i, t in enumerate(best_trials, 1):
        print(f"Configurazione #{i}: value={t.value}, params={t.params}")

if __name__ == '__main__':
    main()
