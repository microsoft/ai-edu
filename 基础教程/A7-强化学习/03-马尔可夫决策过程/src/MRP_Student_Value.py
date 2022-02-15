import Data_Student as ds
import Algorithm_MRP as algoM
import numpy as np


def Student_MonteCarol(gamma):
    episodes = 10000
    end_states = [ds.States.Sleep]
    v = algoM.MonteCarol(ds.Rewards, ds.Matrix, ds.States, end_states, gamma, episodes)
    print("----Monte Carol----")
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))


def InvMatrix(gamma):
    v = algoM.Matrix(ds, gamma)
    print("----Matrix----")
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))
    return v

def Bellman(gamma):
    v = algoM.Bellman(ds.States, ds.Matrix, ds.Rewards, gamma)
    print("----Bellman----")
    for start_state in ds.States:
        print(start_state, "= {:.2f}".format(v[start_state.value]))


if __name__=="__main__":
    gamma = 0.9
    Student_MonteCarol(gamma)
    InvMatrix(gamma)
    Bellman(gamma)
