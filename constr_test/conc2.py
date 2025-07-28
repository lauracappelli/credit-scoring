import numpy as np
#import itertools
#import time
import math

import dimod
import hybrid

i4j2 = [] #(i_1,i_2,i_3,i_4,j_1,j_2)
u4 = [] #(u_1,u_2,u_3,u_4)
i2j1 = [] #(i_1,i_2,j)
u2 = [] #(u_1,u_2)

n=4; m=2; alpha_conc = 0.05
J = n*n*(alpha_conc + (1-alpha_conc)/m)
Jfloor=math.floor(J)
N_S = math.floor(1+math.log2(Jfloor)) # 3 cycles if n=3 and m=2; 4 cycles if n=4 and m=2
mu = 9000

for i1 in range(n):  # 324 cycles if n=3 and m=2; 1024 cycles if n=4 and m=2
	for i2 in range(n):
		for i3 in range(n):
			for i4 in range(n):
				for j1 in range(m):
					for j2 in range(m):
						i4j2.append([i1+1,i2+1,i3+1,i4+1,j1+1,j2+1])
for l in i4j2:
	u_1=(l[0]-1)*m+l[4]
	u_2=(l[1]-1)*m+l[4]
	u_3=(l[2]-1)*m+l[5]
	u_4=(l[3]-1)*m+l[5]
	u4.append([u_1,u_2,u_3,u_4])

for i1 in range(n):  # 18 cycles if n=3 and m=2; 32 cycles if n=4 and m=2
	for i2 in range(n):
		for j in range(m):
			i2j1.append([i1+1,i2+1,j+1])
for l in i2j1:
	u_1=(l[0]-1)*m+l[2]
	u_2=(l[1]-1)*m+l[2]
	u2.append([u_1,u_2])

##############  x   #       w19        #    w'19      #    s
##############  nm  #   (n^4)(m^2)     #  (n^2)m      #   N_S
##############  6   #       324        #     18       #    3
##############  8   #       1024       #     32       #    4
y =            n*m   +   n*n*n*n*m*m   +   n*n*m			+   N_S
print(y)

offset1 = n*m
offset2 = n*m + n*n*n*n*m*m
offset3 = n*m + n*n*n*n*m*m + n*n*m

Q = np.zeros([y,y])
c=0



c += Jfloor*Jfloor

for u in u4:
	if u[0]==u[3]:
		Q[u[0]-1][u[3]-1] += 1
	else:
		Q[u[0]-1][u[3]-1] += 0.5
		Q[u[3]-1][u[0]-1] += 0.5
	if u[1]==u[3]:
		Q[u[1]-1][u[3]-1] += 1
	else:
		Q[u[1]-1][u[3]-1] += 0.5
		Q[u[3]-1][u[1]-1] += 0.5
	if u[2]==u[3]:
		Q[u[2]-1][u[3]-1] += 1
	else:
		Q[u[2]-1][u[3]-1] += 0.5
		Q[u[3]-1][u[2]-1] += 0.5
	if u[0]==u[2]:
		Q[u[0]-1][u[2]-1] += 1
	else:
		Q[u[0]-1][u[2]-1] += 0.5
		Q[u[2]-1][u[0]-1] += 0.5
	if u[1]==u[2]:
		Q[u[1]-1][u[2]-1] += 1
	else:
		Q[u[1]-1][u[2]-1] += 0.5
		Q[u[2]-1][u[1]-1] += 0.5
	if u[0]==u[1]:
		Q[u[0]-1][u[1]-1] += 1
	else:
		Q[u[0]-1][u[1]-1] += 0.5
		Q[u[1]-1][u[0]-1] += 0.5



count = 0
for u in u4:
    Q[offset1+count][offset1+count] += 3
    count+=1

count = 0
for u in u4:
  Q[u[0]-1][offset1+count] -= 1
  Q[offset1+count][u[0]-1] -= 1
  Q[u[1]-1][offset1+count] -= 1
  Q[offset1+count][u[1]-1] -= 1
  Q[u[2]-1][offset1+count] -= 1
  Q[offset1+count][u[2]-1] -= 1
  Q[u[3]-1][offset1+count] -= 1
  Q[offset1+count][u[3]-1] -= 1
  count+=1

for index_s_1 in range(offset3,offset3+N_S):
  for index_s_2 in range(offset3,offset3+N_S):
    if index_s_1==index_s_2:
      Q[index_s_1][index_s_2] += math.pow(2,index_s_1-offset3+index_s_2-offset3)
    else:
      Q[index_s_1][index_s_2] += 0.5*math.pow(2,index_s_1-offset3+index_s_2-offset3)
      Q[index_s_2][index_s_1] += 0.5*math.pow(2,index_s_1-offset3+index_s_2-offset3)

for u in u2:
	if u[0]==u[1]:
		Q[u[0]-1][u[1]-1] -= 2*Jfloor
	else:
		Q[u[0]-1][u[1]-1] -= Jfloor
		Q[u[1]-1][u[0]-1] -= Jfloor

for index_s in range(offset3,offset3+N_S):
  for u in u2:
      Q[u[0]-1][index_s] += math.pow(2,index_s-offset3)
      Q[index_s][u[0]-1] += math.pow(2,index_s-offset3)

for index_s in range(offset3,offset3+N_S):
  for u in u2:
      Q[u[1]-1][index_s] += math.pow(2,index_s-offset3)
      Q[index_s][u[1]-1] += math.pow(2,index_s-offset3)

for index_s in range(offset3,offset3+N_S):
  for u in u2:
    if u[0]==u[1]:
      Q[u[0]-1][u[1]-1] += 2*math.pow(2,index_s-offset3)
    else:
      Q[u[0]-1][u[1]-1] += math.pow(2,index_s-offset3)
      Q[u[1]-1][u[0]-1] += math.pow(2,index_s-offset3)

count = 0
for u in u2:
  for index_s in range(offset3,offset3+N_S):
    Q[offset2+count][offset2+count] += 2*math.pow(2,index_s-offset3)
  count +=1

count = 0
for u in u2:
  for index_s in range(offset3,offset3+N_S):
    Q[offset2+count][index_s] -= math.pow(2,index_s-offset3)
    Q[index_s][offset2+count] -= math.pow(2,index_s-offset3)
  count += 1

count = 0
for u in u2:
  for index_s in range(offset3,offset3+N_S):
    Q[u[0]-1][offset2+count] -= math.pow(2,index_s-offset3)
    Q[offset2+count][u[0]-1] -= math.pow(2,index_s-offset3)
  count += 1

count = 0
for u in u2:
  for index_s in range(offset3,offset3+N_S):
    Q[u[1]-1][offset2+count] -= math.pow(2,index_s-offset3)
    Q[offset2+count][u[1]-1] -= math.pow(2,index_s-offset3)
  count += 1

for index_s in range(offset3,offset3+N_S):
  Q[index_s][index_s] -= 2*Jfloor*math.pow(2,index_s-offset3)

# penality amplification
Q=mu*Q
c=mu*c

"""One counterpart per class"""

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
c += n

print("**")
# -------------------
# ANNEALING APPROACH
# -------------------

# Create the BinaryQuadraticModel
Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1]) if Q[i, j] != 0}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)
# print(bqm)

# Set up the sampler with an initial state
sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=9000000)
state = hybrid.core.State.from_sample({i: 0 for i in range(y)}, bqm)

# Sample the problem
new_state = sampler.run(state).result()

# Estrarre il primo (e spesso unico) campione come lista
result_list = [int(x) for x in new_state.samples.first.sample.values()]

# Ferma il timer
#end_time = time.perf_counter_ns()

print("------------------")
print("ANNEALING APPROACH")
print("------------------")
# print()
# print(new_state)
# print()
# print(new_state.samples)
print(f"\nAnnealer state:\n\t{result_list}")
print(type(result_list))
print(len(result_list)==y)

print(result_list[:n*m])
print(np.array(result_list[:n*m]).reshape(n, m))
print("\nThe matrix is:")
#annealing_matrix = np.array(result_list).reshape(n, m)
#print(annealing_matrix)
print()