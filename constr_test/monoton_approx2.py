import numpy as np
import itertools
import sys
n = 8
m = 3
non_default_prob = 0.70 #(0-1)

def random_d():
  #d = np.random.choice([0, 1], size=n, p=[non_default_prob,1-non_default_prob])
  #print("d:",  d)
  #d = [random.randint(0, 1) for _ in range(n)]
  d =[]
  #ind=0
  for ind in range(n):
    d.append(np.random.choice([0, 1],size=1,p=[1-ind/n,ind/n])[0])
  #print(d)
  #print(np.reshape(d,shape=(n,1)))
  #return np.reshape(d,shape=(n,1))
  return d

if m>n:
  sys.exit(1)

combinations = list(itertools.product([0, 1], repeat=n*m))
matrices_01 = [np.array(combination).reshape(n, m) for combination in combinations]

def R_rate(x,d): #evaluates the sum over i of d i x ij
  vv=[]
  for item in np.transpose(x):
    vv.append(float(np.dot(np.array(d),item)))
  return vv

def func_1(x,d):
  var=[]
  for j in range(m-1):
    sum=0
    for i_1 in range(n):
      for i_2 in range(n):
        sum+=(d[i_1]-d[i_2])*x[i_1,j]*x[i_2,j+1]
    #var.append(bool(sum >= 0))
    #var.append(float(2*n**2+sum))
    #var.append(float(sum))
  return sum

def main():
  d = random_d()
  def_rate = []

  tot_unique = 0
  matrices_unique_1 = []
  for x in matrices_01:
    sum_j = []
    for i in range(n):
      sum=0
      for j in range(m):
        sum += x[i,j]
      sum_j.append(sum)
    sum_i = []
    for j in range(m):
      sum=0
      for i in range(n):
        sum += x[i,j]
      sum_i.append(sum)
    count_o=0; count_s=0;
    for i in range(n-1):
      for j in range(m):
        if x[i,j]+x[i+1,j]==2:
          count_o+=1
    for i in range(n-1):
      for j in range(m-1):
        if x[i,j]+x[i+1,j+1]==2:
          count_s+=1
    #if sum_j == [1]*n and all(item >= 1 for item in sum_i) and x[0,0]==1 and count_o==n-m and count_s==m-1:
    if sum_j == [1]*n and x[0,0]==1 and count_o==n-m and count_s==m-1:
      matrices_unique_1.append(x)
      tot_unique+=1
      def_rate.append(np.divide(R_rate(x,d),x.sum(axis=0)).tolist())
  l=0

  m_func = []
  behavior = []
  for item in matrices_unique_1:
    #print("--------------------------------------------")
    #print("l:", l)
    #print("d:", d)
    #print("dr:",def_rate[l])
    ddef = def_rate[l]
    #print(l,"\n",item,"\n")
    decr =  all(ddef[ll] >= ddef[ll+1] for ll in range(len(ddef) - 1) )
    incr =  all(ddef[ll] <= ddef[ll+1] for ll in range(len(ddef) - 1) )
    if decr == True and incr == True:
      #print("dr is steady (both incr and decr)")
      behavior.append(0)
    if incr == True and decr == False:
      #print("dr is increasing and not steady")
      behavior.append(1)
    if incr == False and decr == True:
      #print("dr is decreasing and not steady")
      behavior.append(2)
    if decr == False and incr == False:
      #print("dr is neither increasing nor decreasing")
      behavior.append(3)
    #print("sum over j=1,m-1 of sum over i_1 i_2 of x[i_1 j] x[i_2 j+1]:")
    #print(func_1(item))
    m_func.append(func_1(item,d))
    l+=1
    #print("--------------------------------------------")
  #print(l)
  #print("behavior")
  #print(behavior)
  #print(len(behavior)==l)
  #print("\n")
  min_func_1 = np.min(m_func)
  print("d:",d)

  # print("0: dr is steady (both incr and decr)")
  # print("1: dr is increasing and not steady")
  # print("2: dr is decreasing and not steady")
  # print("3: dr is neither increasing nor decreasing")

  # print("-----")
  # print("the function sum over j=1,m-1 of sum over i_1 i_2 of x[i_1 j] x[i_2 j+1]")
  # print("has minima corresponding to the configurations:")
  arg_min_func = np.where(min_func_1==m_func)[0].tolist()
  print(arg_min_func)

  #print("the conf.s where dr is steady (both incr and decr)")
  zero_ind = np.where(np.array(behavior) == 0)[0].tolist()
  print(zero_ind)

  #print("the conf.s where dr is increasing and not steady")
  first_ind = np.where(np.array(behavior) == 1)[0].tolist()
  print(first_ind)

  #print("the conf.s where dr is decreasing and not steady")
  second_ind = np.where(np.array(behavior) == 2)[0].tolist()
  print(second_ind)

  #print("the conf.s where dr dr is neither increasing nor decreasing")
  third_ind = np.where(np.array(behavior) == 3)[0].tolist()
  print(third_ind)

  #print("is criterion working?")
  #print(d)
  if all(item in zero_ind+first_ind for item in arg_min_func):
    print("yes\n\n")
    return 1
  else:
    print("no\n\n")
    return 0

# def f_criterion():
#   print(d)
#   if all(item in zero_ind+first_ind for item in arg_min_func):
#     return 1
#   else:
#     return 0

# if __name__ == '__main__':
#     main()
#     main()
#     main()
