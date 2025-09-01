# Credit-scoring

Project repository for the Credit Scoring project in collaboration with Banca Intesa San Paolo for ICSC - Centro Nazionale in HPC, Big Data e Quantum Computing, Spoke 10 - Quantum Computing.

To run the test code, create a Conda environment (or an equivalent python environment) and install [dwave-hybrid](https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/sdk_index.html):

```
conda create -n <nome-env> python=3.13.2
conda activate <nome-env>
pip install dwave-hybrid==0.6.13
pip install pyyaml pandas numpy scipy
```

The `QUBO_formulation` notebook provides the code that builds the cost function and solves the problem using several solver, including the D-Wave annealer simulator. This notebook imports two source files that are located in `src`: 
 * `select_data.py` with all the function to read and manipulate the dataset
 * `check_constraints.py` with the tests for all the constraints


The code that tests classically all the possible combination of n counterparts to m grades is located in `classical_approach.py`. To run the code it is necessary to provide as input the configuration file `config.yaml` with the hyperparameters.
```
python classical_approach.py config.yaml
```

<img src="ICSC-logo.png" width="1000">

<!-- The source code that builds the cost function and solves the problem using the D-Wave simulator is located in the file `cost_function.py`, and the hyperparameters are set in the YAML file `config.yaml`. To run the code, execute the following command within the Conda environment:
```
python cost_function.py config.yaml 
``` -->