import Data_Students2 as ds2
import MDP_Algo_Q as algoQ


if __name__=="__main__":
    gamma = 1
    v = algoQ.Q_pi(ds2.Actions, ds2.Pi_sa, ds2.P_as, ds2.Rewards, gamma)
    for action in ds2.Actions:
        print(action, "= {:.1f}".format(v[action.value]))
