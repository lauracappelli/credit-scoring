import re
import matplotlib.pyplot as plt
import numpy as np
from math import comb
import pandas as pd
import os

def main():

    source = "output/qubo_test/parameter.csv"
    df = pd.read_csv(source, sep=";")
    print(df.head())
    
    qubo_size = []
    energies = []
    correct = []

    for row in df.itertuples(index=False):
        n = row.n
        m = row.m
        shots = row.shots

        for i in range(3):
            # First test
            filename_test1 = f"output/qubo_test/test1_{n}_{m}_run{i}.txt"
            if os.path.isfile(filename_test1):
                with open(filename_test1, "r") as f:
                    # Find the QUBO size
                    for line in f:
                        match_qubo_size = re.search(r"The QUBO problem has (\d+) variables", line)
                        if match_qubo_size:
                            qubo_size.append(int(match.group(1)))
                            break
                    # Find all energies and if the solutions are correct
                    text = f.read()
                    energies.append([float(x) for x in re.findall(r"Energy:\s*([-\d\.eE]+)", text)])
                    correct.append([x == "True" for x in re.findall(r"The solution is correct:\s*(True|False)", text)])
            
            # Second test
            filename_test2 = f"output/qubo_test/test2_{n}_{m}_shots{shots}_run{i}.txt"
            if os.path.isfile(filename_test2):
                with open(filename_test2, "r") as f:
                    # Cerca: dimensione problema, costo della soluzione migliore (o costo medio), numero di sweep

    # Verifica
    print("QUBO size:", qubo_size)
    print("Energy:", energies)
    print("Solution correctness:", correct)

if __name__ == '__main__':
    main()
