import numpy as np

def fun(w,b):
    x = 2*w+3*b
    y=2*b+1
    z=x*y
    return z

def single_variable():
    w = 3
    b = 4
    t = 150
    error = 1e-5
    while(True):
        z = fun(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if delta_z < error:
            break
        delta_b = delta_z /63
        print("delta_b=%f"%delta_b)
        b = b - delta_b

    print("done!")
    print("final b=%f"%b)

def double_variable():
    w = 3
    b = 4
    t = 150
    error = 1e-5
    while(True):
        z = fun(w,b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f"%(w,b,z,delta_z))
        if delta_z < error:
            break
        delta_b = delta_z /63/2
        delta_w = delta_z/18/2
        print("delta_b=%f, delta_w=%f"%(delta_b,delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f"%b)
    print("final w=%f"%w)


if __name__ == '__main__':
    single_variable()
    double_variable()
