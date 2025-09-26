import sys, yaml
import pandas as pd
import numpy as np
import random
from dimod import BinaryQuadraticModel
import os

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

    if data_source == 'ISPdataset':
        datapath = config['data_path']
        if not os.path.exists(datapath):
            print(f"The dataset at '{datapath}' was not found.")
            sys.exit(0)
        return load_data(config)
    
    else:
        # np.random.seed(42)
        default_vec = np.zeros(n)
        
        if data_source == 'random_uniform':
            while np.all(default_vec == 0) or np.all(default_vec == 1):
                default_vec = np.random.choice([0, 1], size = n,p = [1-def_prob, def_prob])

        elif data_source == 'random_incr':
            while np.all(default_vec == 0) or np.all(default_vec == 1):
                default_vec = np.zeros(n)
                for i in range(n):
                    default_vec[i] = np.random.choice([0, 1], p=[1-i/n,i/n])

        elif data_source == 'real_dist':
            random_vec = np.random.rand(n)
            prob = np.linspace(0, 1, n)**2
            index_1 = np.where(random_vec < prob)[0]
            if len(index_1) > def_mod:
                index_1 = np.random.choice(index_1, def_mod, replace=False)
            default_vec[index_1] = 1
        
        elif data_source == 'random_num_def':
            indices_to_set_to_one = random.sample(range(n), def_mod)
            for index in indices_to_set_to_one:
                default_vec[index] = 1

        dataset = pd.DataFrame({
            'counterpart_id': np.arange(1, n+1),
            'default': default_vec,
            'score': -4 + 3 * np.random.randn(n)
        })
        dataset = dataset.sort_values(by='score')

        return dataset

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
