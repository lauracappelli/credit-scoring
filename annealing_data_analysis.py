import os
import re
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def time_vs_sweep_plot(analysis_data, output_dir="output/annealing"):

    sns.set_theme(style="whitegrid")

    for solver in ["classical", "quantum"]:
        df_solver = analysis_data[analysis_data["solver"] == solver]

        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=df_solver,
            x="sweep",
            y="execution_time",
            hue="variables",
            marker="o",
            palette="colorblind",
            linewidth=2,
            markersize=8,
            ax=ax
        )

        solver_title = "Classical Annealing (SA)" if solver == "classical" else "Quantum Annealing (QSA)"

        ax.set_title(f"Time to solution vs. Sweeps - {solver_title}", fontsize=14, pad=12)
        ax.set_xlabel("Sweeps", fontsize=12)
        ax.set_xscale("log")
        ax.set_ylabel("Time to solution (seconds)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", title="QUBO Variables", frameon=True)

        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"time_lineplots_{solver}.pdf")
        plt.savefig(output_filename, format="pdf", bbox_inches="tight")
        output_filename = os.path.join(output_dir, f"time_lineplots_{solver}.png")
        plt.savefig(output_filename, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"Time line plot for {solver.capitalize()} saved to '{output_filename}'")

def solutions_vs_sweep_plot(analysis_data, output_dir="output/annealing"):

    sns.set_theme(style="whitegrid")

    for solver in ["classical", "quantum"]:
        df_solver = analysis_data[analysis_data["solver"] == solver]

        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=df_solver,
            x="sweep",
            y="valid_solutions",
            hue="variables",
            marker="o",
            palette="colorblind",
            linewidth=2,
            markersize=8,
            ax=ax
        )

        solver_title = "Classical Annealing (SA)" if solver == "classical" else "Quantum Annealing (QSA)"

        ax.set_title(f"Valid Solutions vs. Sweeps - {solver_title}", fontsize=14, pad=12)
        ax.set_xlabel("Sweeps", fontsize=12)
        ax.set_xscale("log")
        ax.set_ylabel("Valid Solutions (average on 50 trials)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        
        sns.move_legend(ax,loc="upper left",bbox_to_anchor=(1.02, 1),title="QUBO Variables",frameon=True)
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"lineplots_{solver}.pdf")
        plt.savefig(output_filename, format="pdf", bbox_inches="tight")
        output_filename = os.path.join(output_dir, f"lineplots_{solver}.png")
        plt.savefig(output_filename, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"Line plot for {solver.capitalize()} saved to '{output_filename}'")

def time_sa_vs_qsa_plot(analysis_data, output_filename="output/annealing/time_lineplots_SAvsQSA.pdf"):

    sns.set_theme(style="whitegrid")
    
    grouped = analysis_data.groupby(["variables", "solver", "sweep"])["execution_time"].mean().reset_index()
    unique_vars = sorted(grouped["variables"].unique())
    num_vars = len(unique_vars)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, v in enumerate(unique_vars):
        ax = axes[i]
        for solver in ["classical", "quantum"]:
            sub = grouped[(grouped["variables"] == v) & (grouped["solver"] == solver)]
            if sub.empty:
                continue

            linestyle = "-" if solver == "quantum" else "--"
            marker = "o" if solver == "quantum" else "s"
            color = "#0d47a1" if solver == "quantum" else "#d84315"

            ax.plot(
                sub["sweep"],
                sub["execution_time"],
                label=solver.capitalize(),
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=6
            )

        ax.set_title(f"Variables: {v}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.supxlabel("Number of Sweeps", fontsize=13)
    fig.supylabel("Time to solution (seconds)", fontsize=13)
    fig.suptitle("Time to solution: SA vs. QSA", fontsize=15, y=0.98, fontweight="bold")

    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), title="Solver Type", frameon=True, fontsize=11, title_fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.97, 0.96])

    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.savefig("output/annealing/time_lineplots_SAvsQSA.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Comparative subplots saved to '{output_filename}'")
    
def sa_vs_qsa_plot(analysis_data, output_filename="output/annealing/lineplots_SAvsQSA.pdf"):

    sns.set_theme(style="whitegrid")
    
    grouped = analysis_data.groupby(["variables", "solver", "sweep"])["valid_solutions"].mean().reset_index()
    unique_vars = sorted(grouped["variables"].unique())
    num_vars = len(unique_vars)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, v in enumerate(unique_vars):
        ax = axes[i]
        for solver in ["classical", "quantum"]:
            sub = grouped[(grouped["variables"] == v) & (grouped["solver"] == solver)]
            if sub.empty:
                continue

            linestyle = "-" if solver == "quantum" else "--"
            marker = "o" if solver == "quantum" else "s"
            color = "#0d47a1" if solver == "quantum" else "#d84315"

            ax.plot(
                sub["sweep"],
                sub["valid_solutions"],
                label=solver.capitalize(),
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=6
            )

        ax.set_title(f"Variables: {v}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.supxlabel("Number of Sweeps", fontsize=13)
    fig.supylabel("Mean Valid Solutions Found", fontsize=13)
    fig.suptitle("Valid Solutions: SA vs. QSA", fontsize=15, y=0.98, fontweight="bold")

    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), title="Solver Type", frameon=True, fontsize=11, title_fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.97, 0.96])

    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.savefig("output/annealing/lineplots_SAvsQSA.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Comparative subplots saved to '{output_filename}'")

def overlay_sa_vs_qsa_plot(analysis_data, output_filename="output/annealing/overlay_lineplots_SAvsQSA.pdf"):
    
    sns.set_theme(style="whitegrid")

    grouped = df.groupby(["variables", "solver", "sweep"])[
        ["valid_solutions", "unique_valid_solutions"]
    ].mean().reset_index()

    unique_vars = sorted(grouped["variables"].unique())
    num_vars = len(unique_vars)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, v in enumerate(unique_vars):
        ax = axes[i]
        
        # Valid Solutions in backgound
        for solver in ["classical", "quantum"]:
            sub = grouped[(grouped["variables"] == v) & (grouped["solver"] == solver)]
            if sub.empty:
                continue

            linestyle = "-" if solver == "quantum" else "--"
            marker = "o" if solver == "quantum" else "s"

            ax.plot(
                sub["sweep"],
                sub["valid_solutions"],
                label=f"Valid ({solver.capitalize()}) [Ref]",
                color="#888888",
                linestyle=linestyle,
                marker=marker,
                linewidth=1.5,
                markersize=5,
                alpha=0.4
            )

        # Unique Valid Solutions in foreground
        for solver in ["classical", "quantum"]:
            sub = grouped[(grouped["variables"] == v) & (grouped["solver"] == solver)]
            if sub.empty:
                continue

            linestyle = "-" if solver == "quantum" else "--"
            marker = "o" if solver == "quantum" else "s"
            color = "#0d47a1" if solver == "quantum" else "#d84315"

            ax.plot(
                sub["sweep"],
                sub["unique_valid_solutions"],
                label=f"Unique ({solver.capitalize()})",
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=6
            )

        ax.set_title(f"Variables: {v}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.supxlabel("Number of Sweeps", fontsize=13)
    fig.supylabel("Mean Solutions Found", fontsize=13)
    fig.suptitle("Unique vs. Total Valid Solutions: SA vs. QSA", fontsize=15, y=0.98, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="center left", 
        bbox_to_anchor=(0.98, 0.5), 
        title="Metric & Solver", 
        frameon=True, 
        fontsize=10, 
        title_fontsize=11
    )
    
    plt.tight_layout(rect=[0, 0, 0.97, 0.96])

    png_filename = output_filename.rsplit('.', 1)[0] + ".png"
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.savefig(png_filename, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    print(f"Overlay plot saved to '{output_filename}' and '{png_filename}'")

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

    valid_energy_block_regex = re.compile(
        r"Energy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[\r\n]+\s*The solution is correct:\s*(True|False)"
    )

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

            valid_energies_matches = valid_energy_block_regex.findall(content)
            valid_energies = [
                round(float(e_val), 6) 
                for e_val, is_correct in valid_energies_matches 
                if is_correct == "True"
            ]
            unique_valid_solutions = len(set(valid_energies))

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
                "unique_valid_solutions": unique_valid_solutions,
                "energies": energies,
                "energy_avg": energy_avg,
                "energy_min": energy_min,
                "energy_max": energy_max
            })

    return pd.DataFrame(results)

if __name__ == "__main__":

    df = parse_results(["./output/annealing/quantum", "./output/annealing/classical"])
    
    df_sorted = df.sort_values(by=["solver", "variables", "sweep", "i"])
    cols_to_show = ["solver", "variables", "n", "m", "sweep", "i", "execution_time", "valid_solutions", "unique_valid_solutions"]
    print(df_sorted[cols_to_show].to_string(index=False))

    # Quantum & classical plot: "number of solutions vs sweep"
    solutions_vs_sweep_plot(df)
    sa_vs_qsa_plot(df)

    # Quantum & classical plot: "time vs sweep"
    time_vs_sweep_plot(df)
    time_sa_vs_qsa_plot(df)

    # Quantum & classical plot: "unique solutions vs sweep"
    overlay_sa_vs_qsa_plot(df)