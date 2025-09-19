import numpy as np
import math
import matplotlib.pyplot as plt

n = 10
m = 5

def func_QUBO(n,m):
	if n == 0 and m == 0:
		return   0
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

v_dim_QUBO = []

up = 5

n_vals = []
m_vals = []
# np.arange(1, up)
#m_vals = np.arange(1, up)

for n_item in range(1,up-1):
	for m_item in range(1,up-1):
		if 2 <= m_item < n_item:
			n_vals.append(n_item)
			m_vals.append(m_item)
		else:
			#n_vals.append(float("nan"))
			n_vals.append(0)
			#m_vals.append(float("nan"))
			m_vals.append(0)

# for n_item in n_vals:
# 	for m_item in m_vals:
# 		print([n_item, m_item])

#Z = func_QUBO(n,m)


n_vals_np = np.array(n_vals)
m_vals_np = np.array(m_vals)

# Create meshgrid (pairs of integer coordinates)
N_mesh, M_mesh = np.meshgrid(n_vals_np, m_vals_np)

Z = func_QUBO(N_mesh, M_mesh)

input()



# Define a discrete function f(x, y) = x^2 + y^2 (example)
# Z = X**2 + Y**2

# --- Visualization Option 1: Heatmap ---
plt.figure(figsize=(6, 5))
plt.imshow(Z, origin='lower', extent=(x_vals.min()-0.5, x_vals.max()+0.5,
                                      y_vals.min()-0.5, y_vals.max()+0.5),
           cmap="viridis", aspect="auto")
plt.colorbar(label="f(x, y)")
plt.xticks(x_vals)
plt.yticks(y_vals)
plt.title("Discrete Function f(x,y) = x² + y² (heatmap)")
plt.xlabel("x")
plt.ylabel("y")
print("**")
plt.show()
plt.savefig("figure.png")