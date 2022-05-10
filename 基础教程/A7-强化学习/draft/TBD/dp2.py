@@ -1,24 +0,0 @@
import numpy as np

# C:背包容量，N:物品数量，V:物品价值，S:物品体积

def f(Capacity, Number, Value, Size):
    dp_Value = np.zeros((Number+1, Capacity+1), dtype=np.int32)
    for item in range(1, Number+1):         # 物品
        for capacity in range(1, Capacity+1):   # 容量
            if capacity < Size[item]:           # 容量不够装下这个物品
                dp_Value[item, capacity] = dp_Value[item-1, capacity]   # 仍然使用上一次的结果
            else:                           # 容量足够, 计算新value
                dp_Value[item, capacity] = \
                    max(dp_Value[item-1, capacity], 
                        dp_Value[item-1, capacity - Size[item]] + Value[item])
    
    return dp_Value

if __name__=="__main__":
    Capacity = 5
    Number = 4
    Value = [0,2,4,3,7]
    Size = [0,2,3,5,5]
    result = f(Capacity,Number,Value,Size)
    print(result)