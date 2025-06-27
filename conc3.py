import numpy as np
import itertools
import math

import dimod
import hybrid


# initialization of the parameters
n = 6; m = 3
alpha_conc = 0.05
mu_conc = 10
mu_uniq = 100
gamma = m/(m-1)

i2j1 = [] #(i_1,i_2,j)
u2 = [] #(u_1,u_2)

# initialization of Q and c
Q = np.zeros([n*m, n*m])
c = 0

# add constraint: "first counterpart in first class"
Q[0,0]=-1
c += 1

# add constraint: "one counterpart per class"
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
Q=mu_uniq*Q

for i1 in range(n):  # 18 cycles if n=3 and m=2; 32 cycles if n=4 and m=2
	for i2 in range(n):
		for j in range(m):
			i2j1.append([i1+1,i2+1,j+1])
for l in i2j1:
	u_1=(l[0]-1)*m+l[2]
	u_2=(l[1]-1)*m+l[2]
	u2.append([u_1-1,u_2-1])
     
# add initial cost function: "adjusted herfindhal index"
c += 1/(1-m)
for u_item in u2:
    if u_item[0]==u_item[1]:
      Q[u_item[0]][u_item[1]] += gamma
    else:
      Q[u_item[0]][u_item[1]] += gamma/2

Q=mu_conc*Q

# ---------------------
# BRUTE FORCE APPROACH
# ---------------------

# compute C(Y) = (Y^T)QY + (G^T)Y + c for every Y
Ylist = list(itertools.product([0, 1], repeat=n*m))
Cmin = float('inf')
for ii in range(2**(n*m)):
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