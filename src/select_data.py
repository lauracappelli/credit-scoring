import sys, yaml
import pandas as pd
import numpy as np
import random
from dimod import BinaryQuadraticModel

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
    print('CONFIGURATIONS: ')
    for key in config:
        print(' - ' + key + ': ' + str(config[key]))
        
    # return the config dictionary
    return config

def load_data(config):
    """
    Load the counterparts from the database, selecting only the attributes specified in the configuration file.
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

def generate_data(config):
    """
    Generate a dataset of random counterparts.
    The number of entries is specified in the config file.

    Args:
        dict: iperparameters
    Returns:
        pandas dataframe: dataset of the counterparts ordered by score
    """

    if not isinstance(config['n_counterpart'], (int)):
        sys.exit("Error: specify a number of counterparts")

    # np.random.seed(42)
    def_rand = np.random.choice([0, 1], size=config['n_counterpart'], p=[1-config['default_prob'], config['default_prob']])
    while np.all(def_rand==0) or np.all(def_rand==1):
        def_rand = np.random.choice([0, 1], size=config['n_counterpart'], p=[1-config['default_prob'], config['default_prob']])

    dataset = pd.DataFrame({
        'counterpart_id': np.arange(1, config['n_counterpart']+1),
        'default': def_rand,
        'score': np.random.uniform(-10, 2, size=config['n_counterpart'])
    })

    dataset = dataset.sort_values(by='score')
    return dataset

def generate_data_var_prob(config):
    """
    Generate a dataset of random counterparts.
    The probability of generating the default attribute depends on the score.
    The number of entries is specified in the config file.

    Args:
        dict: hyperparameters
    Returns:
        pandas dataframe: dataset of the counterparts ordered by score
    """

    if not isinstance(config['n_counterpart'], (int)):
        sys.exit("Error: specify a number of counterparts")
        
    n = config['n_counterpart']
    probabilities = np.linspace(0.0, config['default_prob'], n)
    random_values = np.random.rand(n)
    random_booleans = (random_values < probabilities).astype(int)
    # print("Prob:", probabilities)
    # print("Random val:", random_values)
    # print("Default:", random_booleans)

    # np.random.seed(42)
    dataset = pd.DataFrame({
        'counterpart_id': np.arange(1, n+1),
        'default': random_booleans,
        'score': np.random.uniform(-10, 2, size=n)
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
