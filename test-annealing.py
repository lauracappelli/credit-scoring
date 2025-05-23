import numpy as np
import itertools
import time
import math

import dimod
import hybrid
# from hybrid.samplers import TabuProblemSampler
# from hybrid.core import State

# counterparts & classes
n = 6
m = 3
alpha_conc = 0.05

# Avvia il timer
start_time = time.perf_counter_ns()

# Iniatilize Q and c
Q = np.zeros([n*m, n*m])
c = 0

#add penalty: "first counterpart in first class"
for jj in range(1, m):
  Q[jj][jj] += 1
  Q[0][jj] -= 0.5
  Q[jj][0] -= 0.5

#add penalty: "last counterpart in the last class"
# for jj in range(m-1):
#  tt = (n-1)*m+jj
#  Q[tt][tt] += 1
#  Q[(n*m)-1][tt] -= 0.5
#  Q[tt][(n*m)-1] -= 0.5

# add penalty: "one counterpart per class"
for ii in range(n):
  for jj in range(m):
    tt = ii*m+jj
    Q[tt][tt] += -1
  for jj in range(m-1):
    for kk in range(jj+1,m):
      tt = ii*m+jj
      rr = ii*m+kk
      Q[tt][rr] += 1
      Q[rr][tt] += 1
  c += 1

# add penalty: "penalize not permitted submatrix"
for ii in range(n-1):
  for jj in range(m-1):
    aa = ii*m+jj # x_{i,j}=x1
    bb = aa+1   # x_{i,j+1}=x2
    cc = (ii+1)*m+jj # x_{i+1,j}=x3
    dd = cc+1 # x_{i+q,j+1}=x4

    # add linear terms
    Q[aa][aa] += 1
    Q[dd][dd] += 1

    # add quadratic terms
    Q[aa][bb] += 0.5
    Q[bb][aa] += 0.5

    Q[aa][cc] -= 0.5
    Q[cc][aa] -= 0.5

    Q[aa][dd] -= 1
    Q[dd][aa] -= 1

    Q[bb][cc] += 0.5
    Q[cc][bb] += 0.5

    Q[bb][dd] -= 0.5
    Q[dd][bb] -= 0.5

    Q[cc][dd] += 0.5
    Q[dd][cc] += 0.5

# amplify penalties
penality = n*n*n*m*m*m
Q = penality*Q
c = penality*c
print("\nQ:\n", Q)
print("\nc: ", c, "\n")

# ---------------------
# BRUTE FORCE APPROACH
# ---------------------

# compute C(Y) = (Y^T)QY + gY + c for every Y
Ylist = list(itertools.product([0, 1], repeat=n*m))
Cmin = float('inf')
for ii in range(len(Ylist)):
  Y = np.array(Ylist[ii])
  Cy=(Y.dot(Q).dot(Y.transpose()))+c
  if ( Cy < Cmin ):
    Cmin = Cy
    Ymin = Y.copy()

print("--------------------")
print("BRUTE FORCE APPROACH")
print("--------------------")
print("\ncomputing: C(Y) = (Y^T)QY + c")
print("    C(Y) min: ", Cmin)
print("    Y min: ", Ymin)

print("\nThe matrix is:")
matrix = np.array(Ymin).reshape(n, m)
print(matrix)
print()

# -------------------
# ANNEALING APPROACH
# -------------------

# Create the BinaryQuadraticModel
Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1]) if Q[i, j] != 0}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)
# print(bqm)

# Set up the sampler with an initial state
sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=100000)
state = hybrid.core.State.from_sample({i: 0 for i in range(n*m)}, bqm)

# Sample the problem
new_state = sampler.run(state).result()

# Estrarre il primo (e spesso unico) campione come lista
result_list = [int(x) for x in new_state.samples.first.sample.values()]

# Ferma il timer
end_time = time.perf_counter_ns()

print("------------------")
print("ANNEALING APPROACH")
print("------------------")
# print()
# print(new_state)
# print()
# print(new_state.samples)
print(f"\nAnnealer state:\n\t{result_list}")
print("\nThe matrix is:")
annealing_matrix = np.array(result_list).reshape(n, m)
print(annealing_matrix)
print()

# Print size problem & time
elapsed_time_ns = end_time - start_time
print(f"Matrix size:{n*m}*{n*m}")
print(f"Time of emulation: {elapsed_time_ns/10e9} s")




def check_conc(n,m,tt,alpha_conc):
  J = n*n*(alpha_conc + (1-alpha_conc)/m)
  Jfloor=math.floor(J)
  s = 0
  for i1 in range(n):
    for i2 in range(n):
      for j in range(m):
        s = s + tt[i1,j] * tt[i2,j]
  return s <= Jfloor


print(check_conc(n,m,annealing_matrix,alpha_conc))
