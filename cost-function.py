from select_data import *
import dimod
import hybrid
import math

def add_staircase_constr(Q,c,n,m):
	# # add penalty: "first counterpart in first class"
	# for jj in range(1, m):
	# 	Q[jj][jj] += 1
	# 	Q[0][jj] -= 0.5
	# 	Q[jj][0] -= 0.5

	# # add penalty: "last counterpart in the last class"
	# for jj in range(m-1):
	# 	tt = (n-1)*m+jj
	# 	Q[tt][tt] += 1
	# 	Q[(n*m)-1][tt] -= 0.5
	# 	Q[tt][(n*m)-1] -= 0.5

	# add penalty: "one counterpart per class"
	# for ii in range(n):
	# 	for jj1 in range(m):
	# 		for jj2 in range(m):
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

	# # add penalty: "penalize not permitted submatrix"
	# for ii in range(n-1):
	# 	for jj in range(m-1):
	# 		aa = ii*m+jj # x_{i,j}=x1
	# 		bb = aa+1   # x_{i,j+1}=x2
	# 		cc = (ii+1)*m+jj # x_{i+1,j}=x3
	# 		dd = cc+1 # x_{i+q,j+1}=x4

	# 		# add linear terms
	# 		Q[aa][aa] += 1
	# 		Q[dd][dd] += 1

	# 		# add quadratic terms
	# 		Q[aa][bb] += 0.5
	# 		Q[bb][aa] += 0.5

	# 		Q[aa][cc] -= 0.5
	# 		Q[cc][aa] -= 0.5

	# 		Q[aa][dd] -= 1
	# 		Q[dd][aa] -= 1

	# 		Q[bb][cc] += 0.5
	# 		Q[cc][bb] += 0.5

	# 		Q[bb][dd] -= 0.5
	# 		Q[dd][bb] -= 0.5

	# 		Q[cc][dd] += 0.5
	# 		Q[dd][cc] += 0.5
	Q = n*n*n*m*m*Q
	return c
	
def add_concentration_constr(Q,c,n,m,N_S,Jfloor):
	i4j2 = [] #(i_1,i_2,i_3,i_4,j_1,j_2)
	u4 = [] #(u_1,u_2,u_3,u_4)
	i2j1 = [] #(i_1,i_2,j,j)
	u2 = [] #(u_1,u_2,u_3,u_4)
	# print(N_S) # 3 if n=3 and m=2

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


	offset1 = n*m
	offset2 = n*m + n*n*n*n*m*m
	offset3 = n*m + n*n*n*n*m*m + n*n*m*m

	
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

	return c+math.pow(Jfloor,2)

if __name__ == '__main__':
	config = read_config()
	n = config['n_counterpart']
	m = config['m_company']
	alpha_conc = config['alpha_concentration']
	
	J = n*n*(alpha_conc + (1-alpha_conc)/m)
	Jfloor=math.floor(J)
	N_S = math.floor(1+math.log2(Jfloor))
	dim_Q = n*m + n*n*n*n*m*m + n*n*m*m + N_S
	#dim_Q = n*m
	Q = np.zeros([dim_Q,dim_Q])

	c=0
	c=add_staircase_constr(Q,c,n,m)
	print(Q)
	print(c)
	# input()
	# c=add_concentration_constr(Q,c,n,m,N_S,Jfloor)
	# print(Q)
	# print(c)



	Q_dict = {(i, j): Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1]) if Q[i, j] != 0}
	bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict, c)

	sampler = hybrid.samplers.SimulatedAnnealingProblemSampler(num_sweeps=100000)
	state = hybrid.core.State.from_sample({i: 0 for i in range(dim_Q)}, bqm)

	new_state = sampler.run(state).result()

	result_list = [int(x) for x in new_state.samples.first.sample.values()]

	Q_s = result_list[:m*n]
	print(f"\nAnnealer state:\n\t{result_list}")
	print("\nThe matrix is:")
	annealing_matrix = np.array(Q_s).reshape(n, m)
	print(annealing_matrix)
	print()

