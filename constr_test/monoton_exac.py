def monotonicity_constr(m, n, default, offset, mu=1):

    def l_func(v):
        return math.floor(v/(m-1))
    # default vector must be different from all 0 and all 1

    num_of_default = sum(default)
    param = (n-num_of_default)*num_of_default
    Ny = math.floor(1+math.log2(param))
    dim_y = 2*(m-1)*param
    offset_sy = offset + dim_y
    dim_sy = (m-1)*Ny

    dim = offset + dim_y + dim_sy
    Q = np.zeros([dim, dim])

    C_set_minus = []
    C_set = []
    for i1 in range(n):
        for i2 in range(n):
            if default[i1]-default[i2] == -1:
                C_set_minus.append([i1+1,i2+1])
            if default[i1]-default[i2] != 0:
                C_set.append([i1+1,i2+1])

    u2 = []
    for j in range(m-1):
        for item_c_set in C_set:
            u2_1 = (item_c_set[0] -1)*m + (j+1) -1
            u2_2 = (item_c_set[1] -1)*m + (j+1) -1
            u2.append([u2_1 , u2_2])

    u4 = []
    for j in range(m - 1):
        for item_c_set_minus_1 in C_set_minus:
            u_1 = (item_c_set_minus_1[0] - 1) * m + (j + 1) - 1
            u_2 = (item_c_set_minus_1[1] - 1) * m + (j + 1) - 1
            for item_c_set_minus_2 in C_set_minus:
                u_3 = (item_c_set_minus_2[0] - 1) * m + (j + 1) - 1
                u_4 = (item_c_set_minus_2[1] - 1) * m + (j + 1) - 1
                u4.append([u_1, u_2, u_3, u_4])
    
    l2j1 = []
    for l1 in range(Ny):
        for l2 in range(Ny):
            for j in range(m-1):
                l2j1.append([l1,l2,j+1])
    v2 = []
    for l2j1_item in l2j1:
        v_1 = l2j1_item[0]*(m-1) + l2j1_item[2] -1
        v_2 = l2j1_item[1]*(m-1) + l2j1_item[2] -1
        v2.append([v_1,v_2])

    h = []
    for j in range(m-1):
        for c_min_item in C_set_minus:
            u_1=(c_min_item[0]-1)*m + (j+1) -1
            u_2=(c_min_item[1]-1)*m + (j+1) -1
            for l in range(Ny):
                v = l*(m-1) + (j+1) -1
                h.append([u_1,u_2,v])

    for u2_item in u2:
        u_1 = u2_item[0]; u_2 = u2_item[1]
        if u_1==u_2+1:
            Q[u_1,u_2+1] += mu
        else:
            Q[u_1,u_2+1] += mu*0.5
            Q[u_2+1,u_1] += mu*0.5

    for u2_item in u2:
        u_1 = u2_item[0]; u_2 = u2_item[1]
        t=u2.index([u_1,u_2])
        Q[ offset + t, offset + t ] += mu*3

    for u2_item in u2:
        u_1 = u2_item[0]; u_2 = u2_item[1]
        t=u2.index([u_1,u_2])
        Q[u_1,offset + t] += mu*(-2)*0.5
        Q[offset + t,u_1] += mu*(-2)*0.5

    for u2_item in u2:
        u_1 = u2_item[0]; u_2 = u2_item[1]
        t=u2.index([u_1,u_2])
        Q[u_2+1,offset + t] += mu*(-2)*0.5
        Q[offset + t,u_2+1] += mu*(-2)*0.5

    # first summation
    for u4_item in u4:
        u_1 = u4_item[0]; u_2 = u4_item[1]
        u_3 = u4_item[2]; u_4 = u4_item[3]
        t21 = u2.index([u_2,u_1])
        t43 = u2.index([u_4,u_3])
        if t21==t43:
            Q[ offset + t21 , offset + t43 ] += mu
        else:
            Q[ offset + t21 , offset + t43 ] += mu*0.5
            Q[ offset + t43 , offset + t21 ] += mu*0.5
    # second summation
    for u4_item in u4:
        u_1 = u4_item[0]; u_2 = u4_item[1]
        u_3 = u4_item[2]; u_4 = u4_item[3]
        t12 = u2.index([u_1,u_2])
        t34 = u2.index([u_3,u_4])
        if t12==t34:
            Q[ offset + t12 , offset + t34 ] += mu
        else:
            Q[ offset + t12 , offset + t34 ] += mu*0.5
            Q[ offset + t34 , offset + t12 ] += mu*0.5
            
    # first summation
    for v2_item in v2:	
        v_1 = v2_item[0]; v_2 = v2_item[1]
        if v_1==v_2:
            Q[ offset_sy + v_1 , offset_sy + v_2 ] += math.pow( 2, l_func(v_1) + l_func(v_2) )*mu
        else:
            Q[ offset_sy + v_1 , offset_sy + v_2 ] += math.pow( 2, l_func(v_1) + l_func(v_2) )*mu*0.5
            Q[ offset_sy + v_2 , offset_sy + v_1 ] += math.pow( 2, l_func(v_1) + l_func(v_2) )*mu*0.5
    # second summation
    for u4_item in u4:
        u_1 = u4_item[0]; u_2 = u4_item[1]
        u_3 = u4_item[2]; u_4 = u4_item[3]
        t21 = u2.index([u_2,u_1])
        t34 = u2.index([u_3,u_4])
        if t21==t34:
            Q[ offset + t21 , offset + t34 ] += mu*(-2)
        else:
            Q[ offset + t21 , offset + t34 ] += mu*(-2)*0.5
            Q[ offset + t34 , offset + t21 ] += mu*(-2)*0.5

    # first summation
    for h_item in h:
        u_1 = h_item[0]; u_2 = h_item[1]; v = h_item[2]
        t21 = u2.index([u_2,u_1])
        Q[ offset + t21 , offset_sy + v ] += math.pow( 2, l_func(v) + 1 )*mu*0.5
        Q[ offset_sy + v , offset + t21 ] += math.pow( 2, l_func(v) + 1 )*mu*0.5
    # second summation
    for h_item in h:
        u_1 = h_item[0]; u_2 = h_item[1]; v = h_item[2]
        t12 = u2.index([u_1,u_2])
        Q[ offset + t12 , offset_sy + v ] += (-1)*math.pow( 2, l_func(v) + 1 )*mu*0.5
        Q[ offset_sy + v , offset + t12 ] += (-1)*math.pow( 2, l_func(v) + 1 )*mu*0.5

    return Q

    # if config['constraints']['monotonicity'] == True:
    #     (Q_monoton, c_monoton) = monotonicity_constr_appr(m, n, default.T.squeeze(), mu_monotonicity)
    #     #Q_monoton = monotonicity_constr(m, n, default.T.squeeze(), Q.shape[0], mu_monotonicity)
    #     #pad = Q_monoton.shape[0] - Q.shape[0]
    #     #Q = np.pad(Q, pad_width=((0,pad), (0, pad)), mode='constant', constant_values=0) + Q_monoton
    #     Q = Q + Q_monoton
    #     c = c + c_monoton