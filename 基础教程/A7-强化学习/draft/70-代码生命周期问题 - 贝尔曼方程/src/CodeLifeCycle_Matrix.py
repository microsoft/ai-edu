
import numpy as np
import CodeLifeCycle_DataModel as dm


def SolveMatrix(dataModel, gamma):
    # 在对角矩阵上增加一个微小的值来解决奇异矩阵不可求逆的问题
    #I = np.eye(dataModel.N) * (1+1e-7)
    I = np.eye(dataModel.N)
    factor = I - gamma * dataModel.P
    inv_factor = np.linalg.inv(factor)
    vs = np.dot(inv_factor, dataModel.R)
    return vs

if __name__=="__main__":
    dataModel = dm.DataModel()
    v = SolveMatrix(dataModel, 1.0)
    print(v)
    vv = np.around(v,3)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))
