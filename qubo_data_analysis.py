import os
import re
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def solutions_vs_sweep_plot(analysis_data):
    analysis_data = pd.DataFrame(analysis_data)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=analysis_data,
        x="sweep",
        y="valid_solutions",
        hue="variables",
        marker="o",
        palette="viridis",
        linewidth=2,
        markersize=8
    )

    plt.title("Scaling of Valid Solutions vs. Sweeps across Problem Sizes", fontsize=14, pad=12)
    plt.xlabel("Number of Sweeps (Log Scale)", fontsize=12)
    plt.ylabel("Valid Solutions Found", fontsize=12)
    plt.xscale("log")  # Logarithmic scale for sweep values
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="QUBO Variables", frameon=True)
    plt.tight_layout()

    plt.savefig("output/annealing/lineplots.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Line plot saved to '{output_filename}'")

def parse_results(folder_paths):
    results = []
    
    if isinstance(folder_paths, (str, Path)):
        folder_paths = [folder_paths]

    # Regex to extract n, m, sweep, and i from filename: qsa_<n_test>_<n>_<m>_<sweep>_run<i>.txt
    filename_regex = re.compile(r"(qsa|sa)_[^_]+_(\d+)_(\d+)_(\d+)_run(\d+)\.txt")
    reads_regex = re.compile(r"-\s*reads:\s*(\d+)")
    variables_regex = re.compile(r"The QUBO problem has\s*(\d+)\s*variables")
    energy_regex = re.compile(r"Energy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    valid_solutions_regex = re.compile(r"Valid solutions found:\s*(\d+)")
    time_regex = re.compile(r"Time to compute the solution:\s*(\d+(?:\.\d+)?)\s*s")
      
    for folder_path in folder_paths:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Warning: directory '{folder_path}' doesn't exist.")
            continue
            
        for filepath in folder.glob("*.txt"):
            match_filename = filename_regex.match(filepath.name)
            if not match_filename:
                continue
                
            prefix = match_filename.group(1)  # 'qsa' o 'sa'
            n = int(match_filename.group(2))
            m = int(match_filename.group(3))
            sweep = int(match_filename.group(4))
            i = int(match_filename.group(5))

            solver = "quantum" if prefix == "qsa" else "classical"

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            match_reads = reads_regex.search(content)
            reads = int(match_reads.group(1)) if match_reads else 0

            match_vars = variables_regex.search(content)
            variables = int(match_vars.group(1)) if match_vars else None

            match_time = time_regex.search(content)
            execution_time = float(match_time.group(1)) if match_time else None

            energies = [float(e) for e in energy_regex.findall(content)]

            match_valid = valid_solutions_regex.search(content)
            if match_valid:
                valid_solutions = int(match_valid.group(1))
            else:
                valid_solutions = content.count("The solution is correct: True")

            # Calculate energy statistics
            if energies:
                energy_avg = sum(energies) / len(energies)
                energy_min = min(energies)
                energy_max = max(energies)
            else:
                energy_avg = energy_min = energy_max = None

            # Build dictionary for the current file
            results.append({
                "n": n,
                "m": m,
                "sweep": sweep,
                "i": i,
                "reads": reads,
                "variables": variables,
                "solver": solver,
                "execution_time": execution_time,
                "valid_solutions": valid_solutions,
                "energies": energies,
                "energy_avg": energy_avg,
                "energy_min": energy_min,
                "energy_max": energy_max
            })

    return pd.DataFrame(results)

if __name__ == "__main__":

    df = parse_results(["./output/annealing/quantum", "./output/annealing/classical"])
    
    df_sorted = df.sort_values(by=["solver", "variables", "sweep", "i"])
    cols_to_show = ["solver", "variables", "n", "m", "sweep", "i", "execution_time", "valid_solutions"]
    print(df_sorted[cols_to_show].to_string(index=False))

    # Print quantum & classical plot "number of solutions vs sweep"
    # solutions_vs_sweep_plot(analysis_data)