import Data_Students2 as ds2
import Algorithm_MDP_Pi as algoMDP

def Student_V_Pi(gamma):
    v_pi = algoMDP.V_pi(ds2.States, ds2.Pi_sa, ds2.Pr_as, ds2.Rewards, gamma)
    for state in ds2.States:
        print(state, "= {:.1f}".format(v_pi[state.value]))
    return v_pi

def Student_Q_Pi(gamma):
    q_pi = algoMDP.Q_pi(ds2.Actions, ds2.Pi_sa, ds2.Pr_as, ds2.Rewards, gamma)
    for action in ds2.Actions:
        print(action, "= {:.1f}".format(q_pi[action.value]))

def Student_Q_Pi_From_V_Pi(gamma):
    v_pi = algoMDP.V_pi(ds2.States, ds2.Pi_sa, ds2.Pr_as, ds2.Rewards, gamma)
    q_pi = algoMDP.Q_pi_from_V_pi(ds2.Actions, ds2.Pr_as, ds2.Rewards, gamma, v_pi)
    for action in ds2.Actions:
        print(action, "= {:.1f}".format(q_pi[action.value]))

if __name__=="__main__":
    gamma = 1
    Student_V_Pi(gamma)
    Student_Q_Pi(gamma)
    Student_Q_Pi_From_V_Pi(gamma)
