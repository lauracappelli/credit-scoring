import numpy as np
import torch

# --- Configuration ---
n = 10
m = 10
nm = n * m
Jfloor = 3
N_S = 5

# --- Device Setup (for GPU acceleration) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Phase 1: Generate i4j2 and u4 using PyTorch tensor operations ---
# Instead of Python loops, we create ranges and use broadcasting
# This generates all combinations directly as tensors

# Create 1D tensors for each index range
i1_range = torch.arange(1, n + 1, dtype=torch.int64, device=device)
i2_range = torch.arange(1, n + 1, dtype=torch.int64, device=device)
i3_range = torch.arange(1, n + 1, dtype=torch.int64, device=device)
i4_range = torch.arange(1, n + 1, dtype=torch.int64, device=device)
j1_range = torch.arange(1, m + 1, dtype=torch.int64, device=device)
j2_range = torch.arange(1, m + 1, dtype=torch.int64, device=device)

# Use torch.cartesian_prod to get all combinations
# This will be a tensor of shape (n^4 * m^2, 6)
all_combinations = torch.cartesian_prod(
    i1_range, i2_range, i3_range, i4_range, j1_range, j2_range
)

# Now, derive u4 directly from all_combinations
# (l[0]-1)*m+l[4] -> (all_combinations[:,0]-1)*m + all_combinations[:,4]
u_1 = (all_combinations[:, 0] - 1) * m + all_combinations[:, 4]
u_2 = (all_combinations[:, 1] - 1) * m + all_combinations[:, 4]
u_3 = (all_combinations[:, 2] - 1) * m + all_combinations[:, 5]
u_4 = (all_combinations[:, 3] - 1) * m + all_combinations[:, 5]

# Stack them to form the u4 tensor: shape (n^4 * m^2, 4)
u4_tensor = torch.stack([u_1, u_2, u_3, u_4], dim=1)

# Adjust for 0-based indexing for Q matrix
u4_tensor_0_indexed = u4_tensor - 1

# --- Phase 2: Parallelized Q matrix population ---

# Initialize Q as a PyTorch tensor on the chosen device
Q = torch.zeros((nm, nm), dtype=torch.float64, device=device)

# Extract columns for clarity
u0 = u4_tensor_0_indexed[:, 0]
u1 = u4_tensor_0_indexed[:, 1]
u2 = u4_tensor_0_indexed[:, 2]
u3 = u4_tensor_0_indexed[:, 3]

# We need to perform scattered additions.
# torch.scatter_add_ allows parallel updates to specific indices.
# We'll use a temporary tensor of ones to represent the value to add.

# Operations of the form Q[a][b] += value
# Equivalent to torch.scatter_add_(Q, dim=0, index=a, src=value_tensor)
# where value_tensor has shape (len(a), len(b)) or (len(a), 1) depending on how you index.

# Let's group updates for efficiency and clarity.
# Each pair (row_idx, col_idx) gets 0.5 added.
# For symmetric updates (Q[a][b] += val, Q[b][a] += val), we construct two sets of indices.

# Common value for most updates
val_0_5 = torch.full((u0.shape[0],), 0.5, dtype=torch.float64, device=device)
val_0_5_Jfloor = torch.full((u0.shape[0],), 0.5 * (1 - 2 * Jfloor), dtype=torch.float64, device=device)

# --- Group 1: Symmetric additions with 0.5 ---
# Q[u[0]-1][u[3]-1] += 0.5; Q[u[3]-1][u[0]-1] += 0.5
# Q[u[1]-1][u[3]-1] += 0.5; Q[u[3]-1][u[1]-1] += 0.5
# Q[u[2]-1][u[3]-1] += 0.5; Q[u[3]-1][u[2]-1] += 0.5
# Q[u[0]-1][u[2]-1] += 0.5; Q[u[2]-1][u[0]-1] += 0.5
# Q[u[1]-1][u[2]-1] += 0.5; Q[u[2]-1][u[1]-1] += 0.5

# Prepare indices for scatter_add_
# All row indices for these symmetric pairs
all_rows_g1 = torch.cat([u0, u3, u1, u3, u2, u3, u0, u2, u1, u2], dim=0)
# All column indices for these symmetric pairs
all_cols_g1 = torch.cat([u3, u0, u3, u1, u3, u2, u2, u0, u2, u1], dim=0)
# All values to add for these symmetric pairs
all_vals_g1 = torch.cat([val_0_5] * 5 * 2, dim=0) # 5 pairs, each added twice

# Create a flat index for scatter_add_ by mapping (row, col) to a linear index
# This is typically done if you want to use torch.scatter_add_ on a flattened tensor
# or create a sparse tensor. For dense Q, we can do direct updates if we group effectively.

# More straightforward: use advanced indexing with torch.sparse_coo_tensor for cumulative additions
# This is often the most efficient way when you have many overlapping additions to specific indices.

# List of (row_indices, col_indices, values) for sparse tensor construction
sparse_data = []

# Q[u[0]-1][u[3]-1] += 0.5
sparse_data.append((u0, u3, val_0_5))
# Q[u[3]-1][u[0]-1] += 0.5
sparse_data.append((u3, u0, val_0_5))

# Q[u[1]-1][u[3]-1] += 0.5
sparse_data.append((u1, u3, val_0_5))
# Q[u[3]-1][u[1]-1] += 0.5
sparse_data.append((u3, u1, val_0_5))

# Q[u[2]-1][u[3]-1] += 0.5
sparse_data.append((u2, u3, val_0_5))
# Q[u[3]-1][u[2]-1] += 0.5
sparse_data.append((u3, u2, val_0_5))

# Q[u[0]-1][u[2]-1] += 0.5
sparse_data.append((u0, u2, val_0_5))
# Q[u[2]-1][u[0]-1] += 0.5
sparse_data.append((u2, u0, val_0_5))

# Q[u[1]-1][u[2]-1] += 0.5
sparse_data.append((u1, u2, val_0_5))
# Q[u[2]-1][u[1]-1] += 0.5
sparse_data.append((u2, u1, val_0_5))

# Q[u[0]-1][u[1]-1] += 0.5*(1-2 * Jfloor)
sparse_data.append((u0, u1, val_0_5_Jfloor))
# Q[u[1]-1][u[0]-1] += 0.5*(1-2 * Jfloor)
sparse_data.append((u1, u0, val_0_5_Jfloor))

# For the innermost loop: Q[u[0]-1][u[1]-1] += 0.5*pow(2,l)
# This loop needs to be vectorized over 'l' and then added.
# The `pow(2,l)` term can be precomputed as a tensor.
pow2_l_values = torch.pow(2.0, torch.arange(N_S, dtype=torch.float64, device=device))
factor_pow2_l = 0.5 * pow2_l_values.sum() # Sum up all 0.5 * 2^l terms

# Add the sum of 0.5 * 2^l for the specific indices
val_pow2_sum = torch.full((u0.shape[0],), factor_pow2_l, dtype=torch.float64, device=device)
sparse_data.append((u0, u1, val_pow2_sum))
sparse_data.append((u1, u0, val_pow2_sum))

# Combine all sparse data
all_indices_rows = torch.cat([s[0] for s in sparse_data], dim=0)
all_indices_cols = torch.cat([s[1] for s in sparse_data], dim=0)
all_values_to_add = torch.cat([s[2] for s in sparse_data], dim=0)

# Create a sparse tensor from these indices and values
# Note: torch.sparse_coo_tensor handles duplicate indices by summing their values.
indices = torch.stack([all_indices_rows, all_indices_cols], dim=0)
sparse_updates = torch.sparse_coo_tensor(
    indices, all_values_to_add, torch.Size([nm, nm]),
    dtype=torch.float64, device=device
)

# Convert sparse_updates to a dense tensor and add to Q
# This is crucial because Q is a dense tensor.
Q = Q + sparse_updates.to_dense()

print("Q matrix populated successfully!")
# print(Q) # Uncomment to see the Q matrix
print(Q)