import Algorithm_TD as algoTD
import Data_Cliff as data_cliff

if __name__=="__main__":
    env = data_cliff.Env()
    episodes = 10000
    Q1,R1 = algoTD.Saras(env, True, episodes, 0.01, 0.9, None, 10)
    Q2,R2 = algoTD.Q_Learning(env, True, episodes, 0.01, 0.9, None, 10)
    #print(Q1)
    #print(Q2)
    print("Saras")
    algoTD.draw_arrow(Q1)
    print("Q-learning")
    algoTD.draw_arrow(Q2)
