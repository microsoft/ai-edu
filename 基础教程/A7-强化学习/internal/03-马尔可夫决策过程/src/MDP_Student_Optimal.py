import Algorithm_MDP_Star as algoMS
import Data_Students2 as ds2

def Student_V_star(gamma):
    v = algoMS.V_star(ds2.States, ds2.Pi_sa, ds2.Pr_as, ds2.Rewards, gamma)
    for start_state in ds2.States:
        print(start_state, "= {:.1f}".format(v[start_state.value]))

def Student_Q_star(gamma):
    v = algoMS.Q_star(ds2.Actions, ds2.Pi_sa, ds2.Pr_as, ds2.Rewards, gamma)
    for action in ds2.Actions:
        print(action, "= {:.1f}".format(v[action.value]))

def Student_Q_from_V_star(gamma):
    v_star = algoMS.V_star(ds2.States, ds2.Pi_sa, ds2.Pr_as, ds2.Rewards, gamma)
    q_star = algoMS.Q_star_from_V_star(ds2.Actions, ds2.Pr_as, ds2.Rewards, gamma, v_star)
    for action in ds2.Actions:
        print(action, "= {:.1f}".format(q_star[action.value]))

if __name__=="__main__":
    gamma = 1
    Student_V_star(gamma)
    Student_Q_star(gamma)
    Student_Q_from_V_star(gamma)
