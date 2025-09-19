import sys, yaml
import pandas as pd
import numpy as np
import random
from dimod import BinaryQuadraticModel
import os

def check_ISP_dataset(datapath):
    if not os.path.exists(datapath):
        print(f"The dataset at '{datapath}' was not found.")
        #raise FileNotFoundError(f"The dataset at '{datapath}' was not found.")
        sys.exit(0)
    # Your code to load the dataset goes here
    print(f"Dataset successfully loaded from '{datapath}'.")

def read_config():
    """
    Read the configuration file with the iperparameters definition.

    Returns:
        dict: iperparameters 
    """

    # check if the argument exists
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        sys.exit("No configuration file provided")
     
    # read the config file 
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # print all configs
    print('PRINTING ALL THE CONFIGURATIONS: ')
    for key in config:
        print(' - ' + key + ': ' + str(config[key]))
        
    # return the config dictionary
    return config

def print_config():
    print('PRINTING CONFIGURATIONS: ')
    config = read_config()
    n = config['n_counterpart']
    m =  config['grades']
    data_source = config['data_source']
    default_prob = config['default_module']
    default_module = config['default_prob']
    solver = config['solvers']
    shots = config['shots']
    #'constraint' == config['constraints']
    #'mu' == config['mu']
    print("number of counterparts:", n)
    print("number of grades:", m)

    if data_source == 'random_uniform':
        print("generating a random default vector where each element has probabibility of being 1 equal to ", default_prob)
    if data_source == 'random_incr':
        print("generating random default vector where the i-th (zero-based) element has probabibility of being 1 equal to i/n")
    if data_source == 'random_incr':
        print("generating a default vector having d = 1 a number of times equal to", default_module)
    if data_source == 'ISPdataset':
        print("loading the dataset with the following attributes:")
    if solver['brute_force'] == True:
        print("solver: brute force")
    elif solver['dwave_exact'] == True:
        print("solver: exact dwave")
    elif solver['annealing'] == True:
        print("solver: annealing with number of shots equal to", shots)
    elif solver['gurobi'] == True:
        print("solver: gurobi")

    if config['constraints']['one_class'] == True:
        print("mu of uniqueness", config['mu']['one_class'])
    if config['constraints']['logic'] == True:
        print("mu of logical", config['mu']['logical'])
    if config['constraints']['one_class'] == True:
        print("mu of monotonicity", config['mu']['monotonicity'])
    if config['constraints']['concentration'] == True:
        print("mu of uniqueness", config['mu']['concentration'])
    if config['constraints']['min_thr'] == True:
        print("mu of lower card threshold", config['mu']['min_thr'])
    if config['constraints']['max_thr'] == True:
        print("mu of upper card threshold",config['mu']['max_thr'])

def load_data(config):
    """
    Load the counterparts from the ISP database, selecting only the attributes specified in the configuration file.
    The configuration file also specifies the number of entries.

    Args:
        dict: iperparameters
    Returns:
        pandas dataframe: dataset of the counterparts ordered by score
    """
    
    # read dataset
    dataset = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['ID_Fittizio', 'DEFAULT_FLAG_rett_fact', 'score_quant_integrato'])
    dataset = dataset.rename(columns={'ID_Fittizio':'counterpart_id', 'DEFAULT_FLAG_rett_fact':'default', 'score_quant_integrato':'score'})
   
    # select optional attributes
    if config['attributes']['sector'] == 'yes':
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['SETTORE_INT'])
        opt_attr = opt_attr.rename(columns={'SETTORE_INT':'sector'})
        dataset['sector'] = opt_attr['sector'].values

    if config['attributes']['revenue'] == 'yes':
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['classe_indic_dim'])
        opt_attr = opt_attr.rename(columns={'classe_indic_dim':'revenue'})
        dataset['revenue'] = opt_attr['revenue'].values

    if config['attributes']['geo_area'] == 'yes':
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['MACRO_AREA_2b'])
        opt_attr = opt_attr.rename(columns={'MACRO_AREA_2b':'geo_area'})
        dataset['geo_area'] = opt_attr['geo_area'].values

    if config['attributes']['years']:
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['perf_year'])
        opt_attr = opt_attr.rename(columns={'perf_year':'year'})
        dataset['year'] = opt_attr['year'].values
        dataset = dataset[dataset['year'].isin(config['attributes']['years'])]

    # select a specific number of counterparts
    if config['n_counterpart'] != 'all':
        dataset = dataset.sample(n=int(config['n_counterpart']))

    # sort dataset by score
    dataset = dataset.sort_values(by='score')
    return dataset

def generate_or_load_dataset(config):
    """
    According to the given config argument, it generates a dataset of counterparts.
    Args:
        dict: hyperparameters
    Returns:
        pandas dataframe: dataset of the counterparts ordered by score
    """
    n = config['n_counterpart']
    m = config['grades']
    def_prob = config['default_prob']
    def_mod = config['default_module']
    data_source = config['data_source']

    if not isinstance(n, (int)):
        sys.exit("Error: specify the number of counterparts")
    if not 0 < def_prob < 1:
        sys.exit("Error: specify the default probability")
    if not isinstance(def_mod, (int)) or not 1 <= def_mod < n:
        sys.exit("Error: specify a meaningful value of number of defaults")

    if data_source == 'random_uniform':

        # np.random.seed(42)
        def_rand = np.random.choice([0, 1], size = n,p = [1-def_prob, def_prob])
        while np.all(def_rand == 0) or np.all(def_rand == 1):
            def_rand = np.random.choice([0, 1], size = n, p= [1-def_prob, def_prob])
        dataset = pd.DataFrame({
        'counterpart_id': np.arange(1, n+1),
        'default': def_rand,
        'score': np.random.uniform(-10, 2, size = n)
        })
        dataset = dataset.sort_values(by='score')
        return dataset
    elif data_source == 'random_incr':
        # np.random.seed(42)
        def_rand = []
        for ind in range(n):
            def_rand.append(np.random.choice([0, 1],size = n,p=[1-ind/n,ind/n])[0])
        while np.all(def_rand == 0) or np.all(def_rand ==1):
            def_rand = []
            for ind in range(n):
                def_rand.append(np.random.choice([0, 1],size = n,p=[1-ind/n,ind/n])[0])
        dataset = pd.DataFrame({
        'counterpart_id': np.arange(1, n+1),
        'default': def_rand,
        'score': np.random.uniform(-10, 2, size = n)
        })
        dataset = dataset.sort_values(by='score')
        return dataset
    elif data_source == 'random_num_def':
        # np.random.seed(42)
        def_modul = config['default_module']
        def_rand = [0] * n
        indices_to_set_to_one = random.sample(range(n), def_modul)
        for index in indices_to_set_to_one:
            def_rand[index] = 1
        dataset = pd.DataFrame({
        'counterpart_id': np.arange(1, n+1),
        'default': def_rand,
        'score': np.random.uniform(-10, 2, size = n)
        })
        dataset = dataset.sort_values(by='score')
        return dataset
    elif data_source == 'ISPdataset':
        try:
            datapath = config['data_path']
            print(datapath)
            check_ISP_dataset(datapath)
            print("*****")
            return load_data(config)
        except FileNotFoundError as e:
            print("**")
            print(e)

def generate_staircase_matrix(m, n):
    """
    Generate a random binary staircase matrix in which the counterparts are
    equally distributed across in the grades.

    Args:
        m: number of columns (grades)
        n: number of rows (counterparts)
    Returns:
        numpy array: binary staircase matrix
    """

    grad_cardinality = [n//m + 1 if i < n%m else n//m for i in range(m)]
    random.shuffle(grad_cardinality)
    
    matrix = np.zeros([n,m])
    pad = 0
    for i, el in enumerate(grad_cardinality):
        matrix[pad:el+pad,i] = 1
        pad = pad + el
    return matrix

def gen_random_Q(size, c):
    """
    Generate a random binary quadratic problem.

    Args:
        size: linear size of the problem
        c: constant of the binary quadratic problem
    Returns:
        dimod BinaryQuadraticModel: bqm object
    """

    # define binary quadratic problem
    Q_dict = {(i, j): random.randint(0,9) for i in range(size) for j in range(size)}
    bqm = BinaryQuadraticModel.from_qubo(Q_dict, c)

    return bqm
