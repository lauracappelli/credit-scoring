from select_data import *


def add_staircase_constr(Q,c,n,m):
	# add penalty: "first counterpart in first class"
	for jj in range(1, m):
		Q[jj][jj] += 1
		Q[0][jj] -= 0.5
		Q[jj][0] -= 0.5

	# add penalty: "last counterpart in the last class"
	for jj in range(m-1):
		tt = (n-1)*m+jj
		Q[tt][tt] += 1
		Q[(n*m)-1][tt] -= 0.5
		Q[tt][(n*m)-1] -= 0.5

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
	
def add_concentration_constr(Q,c,n,m,alpha_conc):


	
if __name__ == '__main__':
	config = read_config()

	# counterparts & classes
	n = config['n_counterpart']
	m = config['m_company']
	print(n,m)

