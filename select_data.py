import sys, yaml
import pandas as pd
import numpy as np

def read_config():

    # check if the argument was passed
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
    
    # np.random.seed(42)
    dataset = pd.DataFrame({
        'counterpart_id': np.arange(1, config['n_counterpart']+1),
        'default': np.random.choice([0, 1], size=config['n_counterpart'], p=[1-config['default_prob'], config['default_prob']]),
        'score': np.random.uniform(-10, 2, size=config['n_counterpart'])
    })

    dataset = dataset.sort_values(by='score')
    return dataset
    
if __name__ == '__main__':

    config = read_config()

    dataset = generate_data(config) if config['random_data'] == 'yes' else load_data(config)

    attributes = list(dataset.columns.values)
    print("The dataset has {} entries with {} attributes:\n{}".format(dataset.shape[0], len(attributes), attributes))
    print(dataset)