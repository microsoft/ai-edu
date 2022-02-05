import numpy as np

import Data_Student as ds

def InvMatrix(gamma):
    num_state = ds.Matrix.shape[0]
    I = np.eye(num_state)
    tmp1 = I - gamma * ds.Matrix
    tmp2 = np.linalg.inv(tmp1)
    values = np.dot(tmp2, ds.Rewards)
    return values

if __name__=="__main__":
    gamma = 0.9
    v = InvMatrix(gamma)
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))
