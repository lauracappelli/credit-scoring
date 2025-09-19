import numpy as np
import matplotlib.pyplot as plt
import math

def func_QUBO_lin(n,m):
	if not 2 <= m < n:
		return  -30
	else:
		d = math.floor(3*n/100)
		if n < 100:
			#print("setting d = 1 instead of math.floor(3n/100)")
			d = 1
		else:
			#print("setting d = math.floor(3n/100)")
			pass
		# dimensions
		dim_ind = n*m
		# monotonicity
		dim_linear = 2*(m-1)*(n-d)*d
		dim_slack_mon = (m-1)*math.floor( 1 + math.log2( (n-d)*d ) )
		# concentration: no additional variables
		# grade cardinality threshold

		if n < 7:
			#print("n < 7")
			#print("setting low_thr = math.ceil( math.floor(n/m) / 4 ) instead of math.ceil(n/100)")
			#print("setting upp_thr = math.floor( (4/5)*(n - m + 1) ) instead of math.floor(15*n/100)")
			low_thr = math.ceil( math.floor(n/m) / 4 )
			upp_thr = math.floor( (4/5)*(n - m + 1) )
		else:
			#print("setting low_thr = math.ceil(n/100)")
			#print("setting upp_thr = math.floor(15*n/100)")
			low_thr = math.ceil(n/100)
			upp_thr = math.floor(15*n/100)
		# grade cardinality threshold (lower threshold)
		dim_slack_low_th = m*math.floor( 1 + math.log2(n-low_thr) )
		# grade cardinality threshold (upper threshold)
		dim_slack_upp_th = m*math.floor( 1 + math.log2(upp_thr) )
		# QUBO dimension
		dim_QUBO = dim_ind + dim_linear + dim_slack_mon + dim_slack_low_th + dim_slack_upp_th

		#print("*************************************************************")
		#print("number of counterparts:", n)
		#print("number of grades:", m)
		if n < 100:
			#print("number of defaults set equal to 1:", d)
			pass
		else:
			#print("number of defaults math.floor(3*n/100) :", d)
			pass

		#print("grade cardinality lower threshold:", low_thr)
		#print("grade cardinality upper threshold:", upp_thr)
		#print("binary slack vars")
		#print("of the problem:", dim_QUBO)
		#print("1) of basic binary matrix,  nm: ", dim_ind)
		#print("2) related to linear of monoton, 2(m-1)(n-d)d:", dim_linear)
		#print("3) of monoton slack, (m-1) floor of ( 1 + log_2 ((n-d)d) ):", dim_slack_mon)
		if n < 7:
			#print("4) of low thr using low thr = math.ceil( math.floor(n/m) / 4 ):", dim_slack_low_th)
			#print("5) of upp thr using upp thr = math.floor( (4/5)*(n - m + 1) ):", dim_slack_upp_th)
			pass
		else:
			#print("4) of upp thr using low thr = math.ceil(n/100):", dim_slack_low_th)
			pass
			#print("5) of upp thr using upp thr = math.floor(15n/100):", dim_slack_upp_th)

		return dim_QUBO


def func_QUBO_no_lin(n,m):
	if not 2 <= m < n:
		return  -30
	else:
		d = math.floor(3*n/100)
		if n < 100:
			#print("setting d = 1 instead of math.floor(3n/100)")
			d = 1
		else:
			#print("setting d = math.floor(3n/100)")
			pass
		# dimensions
		dim_ind = n*m
		# monotonicity
		dim_slack_mon = (m-1)*math.floor( 1 + math.log2( (n-d)*d ) )
		# concentration: no additional variables
		# grade cardinality threshold

		if n < 7:
			#print("n < 7")
			#print("setting low_thr = math.ceil( math.floor(n/m) / 4 ) instead of math.ceil(n/100)")
			#print("setting upp_thr = math.floor( (4/5)*(n - m + 1) ) instead of math.floor(15*n/100)")
			low_thr = math.ceil( math.floor(n/m) / 4 )
			upp_thr = math.floor( (4/5)*(n - m + 1) )
		else:
			#print("setting low_thr = math.ceil(n/100)")
			#print("setting upp_thr = math.floor(15*n/100)")
			low_thr = math.ceil(n/100)
			upp_thr = math.floor(15*n/100)
		# grade cardinality threshold (lower threshold)
		dim_slack_low_th = m*math.floor( 1 + math.log2(n-low_thr) )
		# grade cardinality threshold (upper threshold)
		dim_slack_upp_th = m*math.floor( 1 + math.log2(upp_thr) )
		# QUBO dimension
		dim_QUBO = dim_ind + dim_slack_mon + dim_slack_low_th + dim_slack_upp_th

		#print("*************************************************************")
		#print("number of counterparts:", n)
		#print("number of grades:", m)
		if n < 100:
			#print("number of defaults set equal to 1:", d)
			pass
		else:
			#print("number of defaults math.floor(3*n/100) :", d)
			pass

		#print("grade cardinality lower threshold:", low_thr)
		#print("grade cardinality upper threshold:", upp_thr)
		#print("binary slack vars")
		#print("of the problem:", dim_QUBO)
		#print("1) of basic binary matrix,  nm: ", dim_ind)
		#print("2) related to linear of monoton, 2(m-1)(n-d)d:", dim_linear)
		#print("3) of monoton slack, (m-1) floor of ( 1 + log_2 ((n-d)d) ):", dim_slack_mon)
		if n < 7:
			#print("4) of low thr using low thr = math.ceil( math.floor(n/m) / 4 ):", dim_slack_low_th)
			#print("5) of upp thr using upp thr = math.floor( (4/5)*(n - m + 1) ):", dim_slack_upp_th)
			pass
		else:
			#print("4) of upp thr using low thr = math.ceil(n/100):", dim_slack_low_th)
			pass
			#print("5) of upp thr using upp thr = math.floor(15n/100):", dim_slack_upp_th)

		return dim_QUBO

m = 10
n_values = np.arange(3, 500)  # n must be > m

y_lin = [func_QUBO_lin(n, m) for n in n_values]
y_no_lin = [func_QUBO_no_lin(n, m) for n in n_values]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(n_values, y_lin, label="QUBO with linear")
plt.xlabel("n")
plt.ylabel("QUBO dimension")
plt.title("func_QUBO_lin (m=10)")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(n_values, y_no_lin, color="orange", label="QUBO without linear")
plt.xlabel("n")
plt.ylabel("QUBO dimension")
plt.title("func_QUBO_no_lin (m=10)")
plt.grid(True)

plt.tight_layout()
plt.savefig("figure2.png")
plt.show()
