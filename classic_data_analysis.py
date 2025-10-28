import re
import matplotlib.pyplot as plt
import numpy as np
from math import comb
import pandas as pd
import os

def main():

    source = "output/classic_test/parameter.csv"
    df = pd.read_csv(source, usecols=[0, 1], names=["n", "m"], header=None)
    
    test_size = []
    test_time = []
    
    for _, row in df.iterrows():
        n = row["n"]
        m = row["m"]

        time_vec = np.zeros([3])
        for i in range(3):
            filename = f"output/classic_test/output_{n}_{m}_run{i}.txt"
            if os.path.isfile(filename):
                with open(filename, "r") as f:
                    for line in f:
                        match = re.search(r"Time\s+([\d.]+)\s*s", line)
                        if match:
                            time_vec[i] = float(match.group(1))
                            break
        test_time.append(np.mean(time_vec))
        test_size.append(comb(n-1, m-1))

    # Print latex table with the results
    df["Problem size"] = test_size
    df["Mean Time"] = test_time
    df.to_latex("output/classic_test/classic_output.tex", index=False)
    
    # Print chart
    plt.scatter(test_size, test_time, color='blue', marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)

    plt.xlabel('Problem size')
    plt.ylabel('Time (s)')
    plt.title('Classical approach - Time vs Problem Size')
    plt.savefig("output/classic_test/classic-plot.png")

if __name__ == '__main__':
    main()
