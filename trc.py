import numpy as np

import torch



n=10; m=10; nm=n*m; Jfloor = 3; N_S = 5; Q = np.zeros([nm,nm])
i4j2 = []
u4 = []

for i1 in range(n):  # 324 cycle if n=3 and m=2
	for i2 in range(n):
		for i3 in range(n):
			for i4 in range(n):
				for j1 in range(m):
					for j2 in range(m):
						i4j2.append((i1+1,i2+1,i3+1,i4+1,j1+1,j2+1))


for l in i4j2:
	u_1=(l[0]-1)*m+l[4]
	u_2=(l[1]-1)*m+l[4]
	u_3=(l[2]-1)*m+l[5]
	u_4=(l[3]-1)*m+l[5]
	u4.append([u_1,u_2,u_3,u_4])

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


print(Q)

def construct_q_matrix(u4, matrix_size, Jfloor, N_S):

    # Initialize Q as a zero matrix of the specified size.
    # We use torch.float32 or torch.float64 for numerical precision.
    # device='cuda' if you want to use GPU, otherwise it defaults to CPU.
    Q = torch.zeros((matrix_size, matrix_size), dtype=torch.float32, device='cuda')
	#Q = torch.zeros((matrix_size, matrix_size))

    # If u4 is not already a PyTorch tensor, convert it.
    # This is often useful for batch processing or if you're mixing data types.
    if not isinstance(u4, torch.Tensor):
        u4 = torch.tensor(u4, dtype=torch.long) # Use torch.long for indices

    # Iterate through each row/entry in u4
    for u in u4:
        # Subtract 1 from indices because Python/PyTorch are 0-indexed
        # while your original code seems to be 1-indexed (u[0]-1, etc.).
        idx0, idx1, idx2, idx3 = u[0] - 1, u[1] - 1, u[2] - 1, u[3] - 1

        # Apply the additions. PyTorch tensors directly support inplace operations (+=)
        # These operations are automatically optimized by PyTorch's backend.
        Q[idx0, idx3] += 0.5
        Q[idx3, idx0] += 0.5
        Q[idx1, idx3] += 0.5
        Q[idx3, idx1] += 0.5
        Q[idx2, idx3] += 0.5
        Q[idx3, idx2] += 0.5
        Q[idx0, idx2] += 0.5
        Q[idx2, idx0] += 0.5
        Q[idx1, idx2] += 0.5
        Q[idx2, idx1] += 0.5

        # Note the Jfloor term
        Q[idx0, idx1] += 0.5 * (1 - 2 * Jfloor)
        Q[idx1, idx0] += 0.5 * (1 - 2 * Jfloor)

        # Inner loop for N_S
        for l in range(N_S):
            power_of_2 = 2**l # Or torch.pow(2, l) for a tensor operation
            Q[idx0, idx1] += 0.5 * power_of_2
            Q[idx1, idx0] += 0.5 * power_of_2

    return Q

print("*")

Q_matrix = construct_q_matrix(u4, nm, Jfloor, N_S)
print("Q Matrix:")
print(Q_matrix)

# Verify a specific element (optional)
# For example, let's look at Q[0,1] (which corresponds to Q[1,2] in your 1-indexed logic)
# for the first entry in u4_data ([1,2,3,4]):
# Q[0,1] gets 0.5*(1-2*Jfloor) from the direct addition.
# And for l=0,1,2, it gets 0.5*2^0, 0.5*2^1, 0.5*2^2.
# So, for the first u: 0.5 * (1 - 2*0.1) + 0.5*1 + 0.5*2 + 0.5*4 = 0.5*0.8 + 0.5 + 1 + 2 = 0.4 + 0.5 + 1 + 2 = 3.9
# If there are multiple entries in u4_data affecting the same Q[i,j], they will sum up.

# For the given example, let's manually trace Q[0,1] based on u4_data:
# From u = [1,2,3,4]:
# Q[0,1] += 0.5 * (1 - 2 * Jfloor_val)
# Q[0,1] += 0.5 * (2**0 + 2**1 + 2**2)  (for N_S_val=3)
# = 0.5 * (1 - 0.2) + 0.5 * (1 + 2 + 4) = 0.5 * 0.8 + 0.5 * 7 = 0.4 + 3.5 = 3.9

# From u = [1,3,5,2] (indices 0,2,4,1):
# Q[0,1] (or Q[1,0]) is affected again.
# Q[idx0, idx1] i.e., Q[0,1]
# Q[idx0, idx1] += 0.5*(1-2 * Jfloor)  -> 0.5*(1-0.2) = 0.4
# For l in range(3):
#   Q[0,1] += 0.5 * pow(2,l) -> 0.5 * (1+2+4) = 3.5
# So from this u, Q[0,1] also gets 0.4 + 3.5 = 3.9

# Total Q[0,1] would be 3.9 + 3.9 = 7.8
# print(f"Q[0,1] calculated: {Q_matrix[0,1].item()}")