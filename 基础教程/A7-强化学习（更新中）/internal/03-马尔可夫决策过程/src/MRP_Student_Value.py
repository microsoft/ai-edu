import Data_Student as ds
import Algorithm_MRP as algoM
import numpy as np


def Student_MonteCarol(ds, gamma):
    episodes = 10000
    end_states = [ds.States.Sleep]
    v = algoM.MonteCarol(ds, end_states, gamma, episodes)
    print("----Monte Carol----")
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))


def InvMatrix(ds, gamma):
    v = algoM.Matrix(ds, gamma)
    print("----Matrix----")
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))
    return v

def Bellman(ds, gamma):
    v = algoM.Bellman(ds, gamma)
    print("----Bellman----")
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))


if __name__=="__main__":
    gamma = 1
    Student_MonteCarol(ds, gamma)
    InvMatrix(ds, gamma)
    Bellman(ds, gamma)
