import math
import numpy as np
uu = [] #(i_1,i_2,i_3,i_4,j_1,j_2)
u = [] #(u_1,u_2,u_3,u_4)
n=3
m=3
for i1 in range(n):
	for i2 in range(n):
		for i3 in range(n):
			for i4 in range(n):
				for j1 in range(m):
					for j2 in range(m):
						uu.append((i1+1,i2+1,i3+1,i4+1,j1+1,j2+1))

for i1 in range(n):
	for i2 in range(n):
		for j in range(m):
			uu.append((i1+1,i2+1,i3+1,i4+1,j1+1,j2+1))


for l in uu: # 324 cycle if n=3 and m=2
	u_1=(l[0]-1)*m+l[4]
	u_2=(l[1]-1)*m+l[4]
	u_3=(l[2]-1)*m+l[5]
	u_4=(l[3]-1)*m+l[5]
	#u.append((u_1,u_2,u_3,u_4))
	u.append([u_1,u_2,u_3,u_4])
print(u)
print("gg")
uuu=u

ii=0
for l in range(len(uu)):
	print(f"uu={uu[l]} - u={u[l]}")
	ii +=1
	if ii==5:
		break

print(len(uu))
print(len(u))

#u_set = set(u)

#print(len(u_set))

print(math.pow(n*m,4))


print(n*n*n*n*m*m)


##############  6   #    324     #
xw = np.zeros( n*m + n*n*n*n*m*m)


print(len(xw)) 

offset = 6

Q = np.zeros([len(xw),len(xw)])

for idu, u in enumerate(uu):
	for a in range(4):
		for b in range(a,4):
			Q[u[a]][u[b]] += 0.5
			Q[u[b]][u[a]] += 0.5
		Q[idu+offset][u[a]]	-= 1
		Q[u[a]][idu+offset]	-= 1
	Q[idu+offset][idu+offset] +=3


print(Q)




def get_unique_vectors_by_permutation(vectors):
  unique_vectors = set()
  for vector in vectors:
    # Convert the vector to a tuple and then to a sorted tuple.
    # Sorting ensures that permutations have the same tuple representation.
    sorted_vector_tuple = tuple(sorted(vector))
    unique_vectors.add(sorted_vector_tuple)
  return unique_vectors

print(uuu)
print("ggg")

vector_set = [[1, 1, 3], [3, 1, 2], [1, 3, 2], [4, 5, 6], [6, 4, 5], [7, 8, 7]]
#vector_set = u
#unique_permutations = get_unique_vectors_by_permutation(vector_set)
unique_permutations = get_unique_vectors_by_permutation(uuu)



#print(f"Original set of vectors: {vector_set}")
print(f"Unique vectors (ignoring permutations): {unique_permutations}")

print(len(unique_permutations))