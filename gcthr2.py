import numpy as np
import math
import itertools
import numpy as np
from cost_function import *

import time
#import dimod
#import hybrid

n=4; m=2

ceil100 = math.ceil(n/100)
floor15 = math.floor(15*n/100)

maxx = 3 #upper threshold (lamda_1)
#maxx = ceil100
minn = 1 #lower threshold (lamda_2)
#minn = floor15
print("counterparts: ",n,"grade: ",m)
print("minimum count for each grade: ",minn)

N_S1 = math.floor(1+math.log2(n-minn))

N_S2 = m*math.floor(1+math.log2(maxx))

#################################  nm  #       N_S1 * m
# n=7; m=4; minn=1; #############  28  +          12 = 48 + 12 = 60
# n=5; m=3; minn=1; #############  15  +        3*3 = 24 (16777216 iterations)
y =                                n*m   	+   N_S1  * m
mu1 = 90
mu2 = 90
print("dimension of the matrix Q:",y,"x",y)
print("numero di variabili slack binarie: ",N_S1*m)

############################ indices of the second term
i2j1 = []; u2 = []
for i1 in range(n):
	for i2 in range(n):
		for j in range(m):
			i2j1.append([i1+1,i2+1,j+1]) #1-based indices
for item in i2j1:
	u_1=(item[0]-1)*m+item[2]
	u_2=(item[1]-1)*m+item[2]
	u2.append([u_1 - 1,u_2 - 1]) #0-based indices

############################ indices of the third term
l2j1_1 = []; su_1 = []
for l1 in range(N_S1):
	for l2 in range(N_S1):
		for j in range(m):
			l2j1_1.append([l1+1,l2+1,j+1]) #1-based indices
for item in l2j1_1:
	ulj_1=(item[0]-1-1)*m+item[2]
	ulj_2=(item[1]-1-1)*m+item[2]
	su_1.append([ulj_1 -1 ,ulj_2 -1]) #0-based indices

############################ indices of the forth term
i1j1 = []; u = []
for i in range(n):
		for j in range(m):
			i1j1.append([i+1,j+1])#1-based indices
for item in i1j1:
	u.append((item[0]-1)*m+item[1]-1)#0-based indices

############################ indices of the fifth and sixth term
lj_1 = []; lj = [];
for l in range(N_S1):
		for j in range(m):
			lj_1.append([l+1,j+1])#1-based indices
for item in lj_1:
	lj.append((item[0]-1-1)*m+item[1]-1) #0-based indices

### recap
#u2 = []		2th term
#su_1 = []	3th term
#u = []			4th term
#lj = []		5th 6th ter

offset = n*m
c=m*minn*minn
Q = np.zeros([y,y]) #y =    n*m   	+   N_S1  * m

for u in u2:
  if u[0]==u[1]:
    Q[u[0]][u[1]] += 1
  else:
    Q[u[0]][u[1]] += 0.5
    Q[u[1]][u[0]] += 0.5

for su in su_1:
  if su[0]==su[1]:
    Q[offset+su[0]][offset+su[1]] += math.pow(2,2*math.ceil((su[0]+1)/m))
  else:
    Q[offset+su[0]][offset+su[1]] += 0.5*math.pow(2,-2+ math.ceil((su[0]+1)/m) + math.ceil((su[1]+1)/m) )
    Q[offset+su[1]][offset+su[0]] += 0.5*math.pow(2,-2+ math.ceil((su[0]+1)/m) + math.ceil((su[1]+1)/m) )

for index_u in u:
    Q[index_u][index_u] -= 2*minn

for sind_1 in lj:
    Q[offset+sind_1][offset+sind_1] += 2*minn*math.pow(2,math.ceil((sind_1+1)/m))

for sind_1 in lj:
  for u_ind in u:
      Q[offset+sind_1][u_ind] += 0.5*(-2)*minn*math.pow(2,math.ceil((sind_1+1)/m))
      Q[u_ind][offset+sind_1] += 0.5*(-2)*minn*math.pow(2,math.ceil((sind_1+1)/m))
c=mu1*c
Q=mu1*Q

#print(Q)

"""Solving QUBO by brute force"""



# -------------------
# ANNEALING APPROACH
# -------------------

# Create the BinaryQuadraticModel
# #Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1]) if Q[i, j] != 0}
# Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1])}
# bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)
# print(len(bqm))

# # Set up the sampler with an initial state
# sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=1000000000)
# state = hybrid.core.State.from_sample({i: 0 for i in range(18)}, bqm)

# # Sample the problem
# new_state = sampler.run(state).result()

# # Estrarre il primo (e spesso unico) campione come lista
# result_list = [int(x) for x in new_state.samples.first.sample.values()]

# annealing_matrix = np.array(result_list)#.reshape(n, m)
# print(len(annealing_matrix))
# print(annealing_matrix)

# Cy_vect = []
# for item in itertools.product([0, 1], repeat=y):
#   Y = np.array(item)
#   Cy=(Y.dot(Q)).dot(Y.transpose())
#   Cy_vect.append(float(Cy))
# #print(Cy_vect)

# Y_vect = []
# i_count=0
# for item in itertools.product([0, 1], repeat=y):
#   if i_count in np.where(Cy_vect == np.min(Cy_vect))[0]:
#     Y = np.array(item)
#     #print(Y)
#     #print(np.array(Y).reshape(n, m)) #[:n, :m]
#     print(np.array(Y))
#     #Y_vect.append(Y.reshape(n, m))
#   i_count +=1
# print(i_count)

# input()


# BQM generation
start_time = time.perf_counter_ns()

bqm = from_matrix_to_bqm(Q, c)
print(len(bqm))
end_time = time.perf_counter_ns()
#print(f"Matrix size:{m*n}*{m*n}")
print(f"Time of generation: {(end_time - start_time)/10e9} s")

# # Solving with annealing 
# start_time = time.perf_counter_ns()  
# result = solveWithAnnealer(m*n, bqm, shots)
# end_time = time.perf_counter_ns()
# result_list = [int(x) for x in result.samples.first.sample.values()]
# annealing_matrix = np.array(result_list).reshape(n, m)
# print(f"\nAnnealing result:\n{annealing_matrix}")    
# print(f"Time of annealing solution: {(end_time - start_time)/10e9} s\n")

# check_staircase(annealing_matrix)
# # check_concentration(annealing_matrix)

# solving exactly
start_time = time.perf_counter_ns()
e_result = exactSolver(bqm)
df_result = e_result.lowest().to_pandas_dataframe()
end_time = time.perf_counter_ns()
elapsed_time_ns = end_time - start_time
print(f"\nALL Exact solutions:\n{df_result}")
# print first result
# matrix = df_result.iloc[:, :m*n].to_numpy()
# # print(f"First solution:\n{matrix[0].reshape(n, m)}")
# # Print all the solutions
# print(f"Exact solutions: {int(matrix.size/(m*n))}")
# for sol in matrix[:]:
#     print(f"solution:\n{sol.reshape(n, m)}")
# print(f"Time of all exact solutions: {elapsed_time_ns/10e9} s")