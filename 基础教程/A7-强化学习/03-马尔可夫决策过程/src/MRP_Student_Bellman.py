import MRP_Algo_Bellman as mbb
import Data_Student as ds

if __name__=="__main__":
    gamma = 0.9
    v = mbb.run(ds.States, ds.Matrix, ds.Rewards, gamma)
    for start_state in ds.States:
        print(start_state, v[start_state.value])
