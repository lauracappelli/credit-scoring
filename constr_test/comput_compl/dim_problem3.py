import numpy as np
import matplotlib.pyplot as plt

# Example function (replace with your QUBO cost function)
def func_QUBO(n, m):
    return n**2 + m**2 - n*m

# Discrete domain
n_vals = range(-5, 6)
m_vals = range(-5, 6)

# Collect results
points = []
values = []
for n in n_vals:
    for m in m_vals:
        points.append((n, m))
        values.append(func_QUBO(n, m))

points = np.array(points)
values = np.array(values)

# --- Visualization as scatter (emphasizes discreteness) ---
plt.figure(figsize=(6, 5))
sc = plt.scatter(points[:, 0], points[:, 1],
                 c=values, s=120, cmap="viridis", edgecolors="k")
plt.colorbar(sc, label="f(n, m)")
plt.xticks(n_vals)
plt.yticks(m_vals)
plt.xlabel("n")
plt.ylabel("m")
plt.title("Discrete function f(n,m) on integer lattice")
plt.grid(alpha=0.3)
plt.show()
plt.savefig("figure.png")
