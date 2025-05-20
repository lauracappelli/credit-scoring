import sys, yaml
import pandas as pd

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

def load_data():
    
    config = read_config()
    
    # read dataset
    dataset = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['ID_Fittizio', 'DEFAULT_FLAG_rett_fact', 'score_quant_integrato'])
   
    # select optional attributes
    if config['attributes']['sector'] == 'yes':
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['SETTORE_INT'])
        dataset['SETTORE_INT'] = opt_attr['SETTORE_INT'].values

    if config['attributes']['revenue'] == 'yes':
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['classe_indic_dim'])
        dataset['classe_indic_dim'] = opt_attr['classe_indic_dim'].values

    if config['attributes']['geo_area'] == 'yes':
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['MACRO_AREA_2b'])
        dataset['MACRO_AREA_2b'] = opt_attr['MACRO_AREA_2b'].values

    if config['attributes']['years']:
        opt_attr = pd.read_csv(open(config['data_path']), delimiter=';', usecols=['perf_year'])
        dataset['perf_year'] = opt_attr['perf_year'].values
        dataset = dataset[dataset['perf_year'].isin(config['attributes']['years'])]

    # select a specific number of counterparts
    if config['n_counterpart'] != 'all':
        dataset = dataset.sample(n=int(config['n_counterpart']))

    return dataset

if __name__ == '__main__':

    dataset = load_data()
    
    dataset_size = dataset.shape[0]
    attributes = list(dataset.columns.values)

    print("The dataset has {} entries with {} attributes:\n{}".format(dataset.shape[0], len(attributes), attributes))
    print(dataset)
