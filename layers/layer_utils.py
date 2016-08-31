from keras.layers import Dense, merge

#Makes Dense connections to a series of previous outputs
#Can be used for making connections to all previous layers
#Eg http://arxiv.org/abs/1608.06993 but for Dense networks
#Avoids the need to concat inputs by product then sum
def make_densedense(output_dim, inputs):
    out_arr = []
    for layer in inputs:
        out_dense = Dense(output_dim)(layer)
        out_arr.append(out_dense)

    if len(out_arr) == 1:
        return out_arr[0]
    else:
        return merge(out_arr, mode='sum')