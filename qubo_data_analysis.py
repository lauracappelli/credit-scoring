import os
import re
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def generate_lineplot_pdf(analysis_data, output_filename="lineplot_valid_solutions_vs_sweeps.pdf"):
    df = pd.DataFrame(analysis_data)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=df,
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

    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Line plot saved to '{output_filename}'")

def parse_results_folder(folder_path):
    results = []
    
    # Regex to extract n, m, sweep, and i from filename: qsa_<n_test>_<n>_<m>_<sweep>_run<i>.txt
    filename_regex = re.compile(r"qsa_[^_]+_(\d+)_(\d+)_(\d+)_run(\d+)\.txt")
    
    # Regex to extract the 'reads' parameter from the configuration section
    reads_regex = re.compile(r"-\s*reads:\s*(\d+)")

    # Regex to extract the number of QUBO variables
    variables_regex = re.compile(r"The QUBO problem has\s*(\d+)\s*variables")
    
    # Regex to extract all Energy values (handles scientific notation as well)
    energy_regex = re.compile(r"Energy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    # Regex to extract the valid solutions count from the summary line
    valid_solutions_regex = re.compile(r"Valid solutions found:\s*(\d+)")

    folder = Path(folder_path)
    
    for filepath in folder.glob("qsa_*.txt"):
        match_filename = filename_regex.match(filepath.name)
        if not match_filename:
            continue
            
        n = int(match_filename.group(1))
        m = int(match_filename.group(2))
        sweep = int(match_filename.group(3))
        i = int(match_filename.group(4))

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract 'reads'
        match_reads = reads_regex.search(content)
        reads = int(match_reads.group(1)) if match_reads else 0

        # Extract 'variables'
        match_vars = variables_regex.search(content)
        variables = int(match_vars.group(1)) if match_vars else None

        # Extract energy values
        energies = [float(e) for e in energy_regex.findall(content)]

        # Extract valid solutions count
        match_valid = valid_solutions_regex.search(content)
        if match_valid:
            valid_solutions = int(match_valid.group(1))
        else:
            # Fallback in case the summary line is missing: count occurrences of "The solution is correct: True"
            valid_solutions = content.count("The solution is correct: True")

        # Calculate energy statistics
        if energies:
            energy_avg = sum(energies) / len(energies)
            energy_min = min(energies)
            energy_max = max(energies)
        else:
            energy_avg = energy_min = energy_max = None

        # Build dictionary for the current file
        file_data = {
            "n": n,
            "m": m,
            "sweep": sweep,
            "i": i,
            "reads": reads,
            "variables": variables,
            "valid_solutions": valid_solutions,
            "energies": energies,
            "energy_avg": energy_avg,
            "energy_min": energy_min,
            "energy_max": energy_max
        }

        results.append(file_data)

    return results

if __name__ == "__main__":

    folder_path = "./output/annealing/quantum"   
    analysis_data = parse_results_folder(folder_path)
    
    # Print sorted data
    sorted_data = sorted(
        analysis_data, 
        key=lambda x: (x["variables"] if x["variables"] is not None else float("inf"), x["sweep"])
    )
    for item in sorted_data:
        print(
            f"n: {item['n']} | "
            f"m: {item['m']} | "
            f"variables: {item['variables']} | "
            f"sweep: {item['sweep']} | "
            f"i: {item['i']} | "
            f"valid_solutions: {item['valid_solutions']}"
        )

    # 3. Generate and save PDF plots
    generate_heatmap_pdf(sorted_data, "heatmap.pdf")
    generate_lineplot_pdf(sorted_data, "lineplot.pdf")