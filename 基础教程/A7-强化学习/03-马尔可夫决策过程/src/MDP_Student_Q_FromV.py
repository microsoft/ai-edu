import Data_Students2 as ds2
import MDP_Algo_V as algoV
import MDP_Algo_Q_FromV as algoQ

if __name__=="__main__":
    gamma = 1
    v = algoV.V_pi(ds2.States, ds2.Pi_sa, ds2.P_as, ds2.Rewards, gamma)
    print(v)
    v = algoQ.Q2_pi(ds2.Actions, ds2.P_as, ds2.Rewards, gamma, v)
    for action in ds2.Actions:
        print(action, "= {:.1f}".format(v[action.value]))
