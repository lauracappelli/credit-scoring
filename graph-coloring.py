import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import itertools
import dimod
import hybrid
import time

def test_selector(id):

  if (id == 1):
    # form paper QUBO
    N=3; K=2
    G=np.zeros([N,N])
    G[0][1]=1
    G[1][0]=1; G[1][2]=1
    G[2][1]=1

  elif (id == 2):
    N=5; K=3
    G=np.zeros([N,N])
    G[0][1]=1; G[0][4]=1
    G[1][0]=1; G[1][2]=1; G[1][3]=1; G[1][4]=1
    G[2][1]=1; G[2][3]=1
    G[3][1]=1; G[3][2]=1; G[3][4]=1
    G[4][0]=1; G[4][1]=1; G[4][3]=1

  elif (id == 3):
    N=4; K=2
    G=np.zeros([N,N])
    G[0][1]=1
    G[1][0]=1; G[1][2]=1; G[1][3]=1
    G[2][1]=1
    G[3][1]=1

  elif (id == 4):
    N=4; K=2
    G=np.zeros([N,N])
    G[0][1]=1; G[0][3]=1
    G[1][0]=1; G[1][2]=1
    G[2][1]=1; G[2][3]=1
    G[3][0]=1; G[3][2]=1

  elif (id == 5):
    N=5; K=2
    G=np.zeros([N,N])
    G[0][1]=1
    G[1][0]=1; G[1][2]=1
    G[2][1]=1; G[2][3]=1
    G[3][2]=1; G[3][4]=1
    G[4][3]=1

  elif (id == 6):
    # from paper flight assignment
    N=6; K=3
    G=np.zeros([N,N])
    G[0][2]=1
    G[1][3]=1; G[1][4]=1
    G[2][0]=1; G[2][4]=1
    G[3][1]=1; G[3][4]=1; G[3][5]=1
    G[4][1]=1; G[4][2]=1; G[4][3]=1
    G[5][3]=1

  elif (id == 7):
    # from paper frequency allocation
    N=6; K=3
    G=np.zeros([N,N])
    G[0][1]=1; G[0][5]=1
    G[1][0]=1; G[1][2]=1; G[1][3]=1; G[1][5]=1
    G[2][1]=1; G[2][3]=1
    G[3][1]=1; G[3][2]=1; G[3][4]=1
    G[4][3]=1; G[4][5]=1
    G[5][0]=1; G[5][1]=1; G[5][4]=1

  elif (id == 8):
    N=2; K=2
    G=np.zeros([N,N])
    G[0][1]=1
    G[1][0]=1

  return G, N, K

def generate_random_graph(num_nodes, edge_percentage, K):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Numero massimo di archi in un grafo non orientato senza loop
    max_edges = num_nodes * (num_nodes - 1) // 2  
    num_edges = int((edge_percentage / 100) * max_edges) 

    # Genera tutte le possibili coppie di nodi e le mescola casualmente
    possible_edges = list(itertools.combinations(range(num_nodes), 2))
    random.shuffle(possible_edges)

    # Aggiunta controllata degli archi rispettando il vincolo di grado < K
    selected_edges = []
    node_degrees = {node: 0 for node in range(num_nodes)}

    for u, v in possible_edges:
        if node_degrees[u] < K-1 and node_degrees[v] < K-1:
            selected_edges.append((u, v))
            node_degrees[u] += 1
            node_degrees[v] += 1
            if len(selected_edges) >= num_edges:
                break  # Fermati quando hai raggiunto il numero richiesto di archi

    G.add_edges_from(selected_edges)

    # Convertiamo la matrice di adiacenza in un array NumPy
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    
    return adjacency_matrix, num_nodes, K

def draw_graph(G, color_list,filename):
  fig, ax = plt.subplots(figsize=(20, 12))
  if len(color_list) == 0:
    nx.draw_circular(G, with_labels=True, font_weight='bold')
  else:
    nx.draw_circular(G, with_labels=True, font_weight='bold', node_color=color_list)
  plt.savefig(filename)
  plt.close()

def getColors(solution,N,K):
  colors = np.empty(N)
  for ii in range(len(solution)):
    if solution[ii] == 1:
      inode = (ii//K)
      icolor = (ii%K)
      colors[inode] = icolor
  return colors

def check_result(G, solution, N, K):
  colors = getColors(solution, N, K)

  for i in range(N):
    for j in range(N):
      if G[i][j] != 0:
        if(colors[i]==colors[j]):
          print("KO!")
          exit(1)

# define: the problem where
# N: n. of node, K: n. of color, G: adjacency matrix
testid = 7
#G, N, K = test_selector(testid)
G, N, K = generate_random_graph(32, 30, 4)
print("------ INPUT -------")
print("N: ", N)
print("K: ", K)
print("G:\n", G)
print()
draw_graph(nx.from_numpy_array(G), [], "ch0.png")

# Iniatilize Q assuming the mapping
# (i,k) --> (i*K+k), i index of node, k index of color
Q = np.zeros([N*K, N*K])
g = np.zeros(N*K)
c = 0

# add penalties from nodes constraints
for ii in range(N):
  for kk in range(K):
    tt = ii*K+kk
    g[tt] = -1
  for kk in range(K-1):
    for kkk in range(kk+1,K):
      tt = ii*K+kk
      rr = ii*K+kkk
      Q[tt][rr] += 1
      Q[rr][tt] += 1
      # print("kk: ", kk, "ii: ", ii, "kkk: ", kkk, "tt: ", tt, "rr: ",rr)
  c += 1
# print("Q:\n", Q)

# add penalties from edges constraints
for ii in range(N):
  for jj in range(ii+1,N):
    if G[ii][jj] == 1:
      for kk in range(K):
        tt = ii*K+kk
        rr = jj*K+kk
        Q[tt][rr] += 0.5
        Q[rr][tt] += 0.5
        # print("ii: ", ii, "jj: ", jj, "kk: ", kk, "tt: ", tt, "rr: ", rr)
# print("Q:\n", Q)

# Step 4
penality = 4
Q = penality*Q
g = penality*g
c = penality*c
print("Q:\n", Q)
print("\ng: ", g)
print("\nc: ", c, "\n")

# ---------------------
# BRUTE FORCE APPROACH
# ---------------------

# # compute C(Y) = YQY + gY + c for every Y
# Ylist = list(itertools.product([0, 1], repeat=(N*K)))
# Cmin = float('inf')
# for ii in range(len(Ylist)):
#   Y = np.array(Ylist[ii])
#   Cy=(Y.dot(Q).dot(Y.transpose()))+g.dot(Y.transpose())+c
#   if ( Cy < Cmin ):
#     Cmin = Cy
#     Ymin = Y.copy()

# print("--------------------")
# print("BRUTE FORCE APPROACH")
# print("--------------------")
# print("\ncomputing: C(Y) = YQY + c")
# print("    C(Y) min: ", Cmin)
# print("    Y min: ", Ymin)
# print("\nGraph coloring solution with QUBO:")
# for ii in range(len(Ymin)):
#   if Ymin[ii] == 1:
#     print("y_%02d = x_%d%d = true --> N%d C%d" % (ii, (ii//K), (ii%K), (ii//K), (ii%K)))
# print()
# draw_graph(nx.from_numpy_array(G), getColors(Ymin, N, K), "ch1.png")

# -------------------
# ANNEALING APPROACH
# -------------------

# Start timer
start_time = time.perf_counter_ns()

# Create the BinaryQuadraticModel 
Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1]) if Q[i, j] != 0}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)
bqm.add_linear_from_array(g)
# print(bqm)

# Set up the sampler with an initial state
sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=10000)
state = hybrid.core.State.from_sample({i: 0 for i in range(N*K)}, bqm)

# Sample the problem
new_state = sampler.run(state).result()

# Extraxt the first result
result_list = [int(x) for x in new_state.samples.first.sample.values()]

# Stop timer
end_time = time.perf_counter_ns()

print("------------------")
print("ANNEALING APPROACH")
print("------------------")
# print()
# print(new_state)
# print()
# print(new_state.samples)
print(f"\nAnnealer state:\n\t{result_list}")
print("\nGraph coloring solution with annealer:")
for ii in range(len(result_list)):
  if result_list[ii] == 1:
    print(f"x_{ii//K}{ii%K} = true --> N{ii//K} C{ii%K}")
print()
draw_graph(nx.from_numpy_array(G), getColors(result_list, N, K), "ch2.png")

# Print size problem & time
elapsed_time_ns = end_time - start_time
print(f"Matrix size:{N*K}*{N*K}")
print(f"Time of emulation: {elapsed_time_ns/10e9} s")

check_result(G, result_list, N, K)