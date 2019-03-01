import numpy as np
from pathlib import Path

def LoadData():
    Xfile = Path("HousePriceXData.dat")
    Yfile = Path("HousePriceYData.dat")
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

if __name__ == '__main__':
    X,Y = LoadData()
    num_example = X.shape[1]
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[:,0:num_example]).T))
    y = (Y[:,0:num_example]).T

    a = np.dot(x.T, x)
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, y)
    print(e)
    print("w1=%f,w2=%f,w3=%f,b=%f"%(e[1],e[2],e[3],e[0]))
