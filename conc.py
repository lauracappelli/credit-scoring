import math
import itertools
import numpy as np

import dimod
import hybrid

i4j2 = [] #(i_1,i_2,i_3,i_4,j_1,j_2)
u4 = [] #(u_1,u_2,u_3,u_4)
i2j1 = [] #(i_1,i_2,j,j)
u2 = [] #(u_1,u_2,u_3,u_4)

n=6; m=3; alpha_conc = 0.05
J = n*n*(alpha_conc + (1-alpha_conc)/m)
Jfloor=math.floor(J)
N_S = math.floor(1+math.log2(Jfloor))
print(N_S) # 3 if n=3 and m=2

for i1 in range(n):  # 324 cycle if n=3 and m=2
	for i2 in range(n):
		for i3 in range(n):
			for i4 in range(n):
				for j1 in range(m):
					for j2 in range(m):
						i4j2.append((i1+1,i2+1,i3+1,i4+1,j1+1,j2+1))

for i1 in range(n):
	for i2 in range(n):
		for j in range(m):
			i2j1.append((i1+1,i2+1,j+1,j+1))

for l in i4j2:
	u_1=(l[0]-1)*m+l[4]
	u_2=(l[1]-1)*m+l[4]
	u_3=(l[2]-1)*m+l[5]
	u_4=(l[3]-1)*m+l[5]
	u4.append([u_1,u_2,u_3,u_4])

for l in i2j1:
	u_1=(l[0]-1)*m+l[0]
	u_2=(l[1]-1)*m+l[0]
	u2.append([u_1,u_2])

##############  x   #       w19        #    w'19      #    s
##############  nm  #   (n^4)(m^2)     #  (n^2)(m^2)  #   N_S
##############  6   #       324        #     36       #    3
y = np.zeros( n*m   +   n*n*n*n*m*m    +  n*n*m*m     +  N_S    )
print(len(y))   #  369


offset1 = n*m
offset2 = n*m + n*n*n*n*m*m
offset3 = n*m + n*n*n*n*m*m + n*n*m*m

Q = np.zeros([len(y),len(y)])

#Q = np.full([len(y), len(y)], float('nan'))

for u in u4:
	Q[u[0]-1][u[3]-1] += 0.5
	Q[u[3]-1][u[0]-1] += 0.5
	Q[u[1]-1][u[3]-1] += 0.5
	Q[u[3]-1][u[1]-1] += 0.5
	Q[u[2]-1][u[3]-1] += 0.5
	Q[u[3]-1][u[2]-1] += 0.5
	Q[u[0]-1][u[2]-1] += 0.5
	Q[u[2]-1][u[0]-1] += 0.5
	Q[u[1]-1][u[2]-1] += 0.5
	Q[u[2]-1][u[1]-1] += 0.5
	Q[u[0]-1][u[1]-1] += 0.5*(1-2 * Jfloor)
	Q[u[1]-1][u[0]-1] += 0.5*(1-2 * Jfloor)
	for l in range(N_S):
		Q[u[0]-1][u[1]-1] += 0.5*pow(2,l)
		Q[u[1]-1][u[0]-1] += 0.5*pow(2,l)

for ind in range(n*n*n*n*m*m):
	for l in range(N_S):
		Q[ind+offset1-1][ind+offset1-1] += 3*pow(2,l+1)

for ind in range(n*n*m*m):
	Q[ind+offset2-1][ind+offset2-1] += 0.5

for ind in range(N_S):
	for l_1 in range(N_S):
		for l_2 in range(N_S):
			Q[ind+offset3-1][ind+offset3-1] += 0.5*pow(2,l_1 + l_2)

for ind, u in enumerate(u4):
	Q[u[0]][ind+offset1] -=1
	Q[ind+offset1][u[0]] -=1
	Q[u[1]][ind+offset1] -=1
	Q[ind+offset1][u[1]] -=1
	Q[u[2]][ind+offset1] -=1
	Q[ind+offset1][u[2]] -=1
	Q[u[3]][ind+offset1] -=1
	Q[ind+offset1][u[3]] -=1

for ind, u in enumerate(u2):
	Q[u[0]][ind+offset2] -= 0.5
	Q[ind+offset2][u[0]] -= 0.5
	Q[u[1]][ind+offset2] -= 0.5
	Q[ind+offset2][u[1]] -= 0.5

for ind in range(N_S):
	for ind2, u in enumerate(u2):
		Q[u[0]][ind+offset3] += pow(2,l)
		Q[ind+offset3][u[0]] += pow(2,l)
		Q[u[1]][ind+offset3] += pow(2,l)
		Q[ind+offset3][u[1]] += pow(2,l)

for ind in range(N_S):
	for ind2, u in enumerate(u2):
		Q[u[0]+offset2][ind+offset3] += pow(2,l)
		Q[ind+offset3][u[0]+offset2] += pow(2,l)

c=0
# Create the BinaryQuadraticModel
Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1]) if Q[i, j] != 0}
bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)
# print(bqm)

print("**")

# Set up the sampler with an initial state
sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=100)
state = hybrid.core.State.from_sample({i: 0 for i in range(len(y))}, bqm)

# Sample the problem
new_state = sampler.run(state).result()

# Estrarre il primo (e spesso unico) campione come lista
result_list = [int(x) for x in new_state.samples.first.sample.values()]


def check_conc(n,m,tt,alpha_conc):
  J = n*n*(alpha_conc + (1-alpha_conc)/m)
  Jfloor=math.floor(J)
  s = 0
  for i1 in range(n):
    for i2 in range(n):
      for j in range(m):
        s = s + tt[i1,j] * tt[i2,j]
  return s <= Jfloor





#Ylist_c = list(itertools.product([0, 1], repeat=(len(n*m))))

#print(check_conc(n,m,Ylist_c,alpha_conc))
