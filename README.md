# Credit-scoring

<img src="ICSC-logo.png" width="1000">

Repository of the Quantum Credit Scoring project, realized and developed within the context of **ICSC** - Centro Nazionale in HPC, Big Data e Quantum Computing, Spoke 10 - Quantum Computing.

This project, realized in collaboration with **Banca Intesa Sanpaolo**, aims to develop an algorithm that uses quantum computing features to solve the financial problem of the credit rating scale definition. More in details, the goal is to categorize $n$ counterparts into $m$ grades according to several constraints. 

At the beginning of the collaboration, we implement the notebook `read_dataset.ipynb`. This notebook reads the dataset of the counterpats provided by Intesa Sanpaolo and performs some statistical analyses on it. Note: The dataset cannot be shared; for demonstration purposes, for the orher script we implemented functions that generate random datasets.

The code that solves the rating scale definition problem requires some Python packages, such as [dwave-hybrid](https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/sdk_index.html). We used a Conda environment created as follows:

```
conda create -n <env_name> python=3.13.2
conda activate <env_name>
pip install pyyaml pandas numpy scipy dwave-hybrid==0.6.13
```

The `QUBO_formulation.ipynb` notebook provides the code that solves the QUBO-formulated problem on small instances. It uses several solvers, including the D-Wave annealer simulator. This notebook imports two source files that are located in the `src` folder: 
 * `select_data.py`, which defines all the functions to read and manipulate the dataset;
 * `check_constraints.py`, which contains the tests to check if the constraints are fulfilled.

To run the solvers on significant-size problems, the QUBO formulation is provided also as a Python script in the file `cost_function.py`. To set up the hyperparameters in this case, we use the YAML file `config.yaml`.
To execute the code, run the following command inside the Conda environment:
```
python cost_function.py config.yaml 
```

The code that tests all possible combinations of $n$ counterparts to $m$ grades classically (with a brute force approach) is located in `classical_approach.py`. This script requires the same YAML file as before, which contains the hyperparameter definitions.
```
python classical_approach.py config.yaml
```