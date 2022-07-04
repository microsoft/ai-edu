
## 10.2 

### 10.2.1 温故知新

#### 充分利用样本数据

在 6.3 节中，我们根据回报 $G$ 的定义：

$$
G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+ \cdots +\gamma^{T-t-1} R_{T}
\tag{10.2.1}
$$

以及价值函数 $V$ 的定义：

$$
v_t(s) = \mathbb E [G_t | S_t = s]
\tag{10.2.2}
$$

然后使用最朴素的蒙特卡洛采样法初步计算出了安全驾驶问题的各个状态的价值函数。简述其过程如下：

1. 使用蒙特卡洛采样，每次采样都需要指定一个初始状态 $s_0$，然后开始在幕内循环；
2. 得到一个 $R_t$ 后，计算 $G = G + \gamma^{t-1}R_t$，直到幕内循环结束；
3. 进行下一次采样，$G$ 值清零后再用 step 2 的循环重新计算 $G$ 值并累积；
4. 最后求 $G$ 的平均（数学期望）得到 $v(s_0)$，即式（10.2.2）。

这种方法的特点是：
1. 稳定；
2. 省空间，不需要记录中间状态，来了一个 $R_t$ 后立刻计算，然后扔掉，只保留 $G$ 值；
3. 随着幕数的增加，会更逼近真实值；
4. 每次只计算一个状态；
4. 遍历状态并在环境控制下重复多次，速度慢；
5. 浪费了中间的采样结果。

聪明的读者可能会发现一个问题：如果不“从头”开始，而是从第二个、第三个状态开始计算，是不是就能在一次采样中就可以得到很多状态的 G 值呢？

<center>
<img src='./img/MC-1.png'>

图 10.2.1 
</center>


如图 10.2.1 所示，从 $S_0$ 开始一幕的采样，到 $T$ 为止结束，得到 $R_1,\cdots,R_T$ 的序列后：
- 第一行：固然可以从 $R_1$ 开始计算出 $G_0$。
- 第二行：但是如果从 $R_2$ 开始，不就能计算出 $G_1$ 了吗？
- 第三行：同理，还可以计算出这一采样序列中的任意的 $G_t$ 出来。

这样的话利用一次采样结果可以计算出很多状态的 $G$ 值，会大幅提高算法的效率。


#### 贝尔曼方程

在第 6 章时，我们还没有学习贝尔曼方程（第 7 章的内容），所以当时使用了蒙特卡洛法依靠大量的采样来计算 $G$ 值。由于安全驾驶问题实际上是有一个模型的，所以可以利用贝尔曼方程得到该问题的“近似真实解”，以此为基准（Ground Truth），来衡量算法的性能。而性能又包括两个方面：1. 速度；2. 准确度。需要在这二者之间寻找平衡。

下面是用贝尔曼方程的矩阵法计算出来的状态价值函数值。

【代码位置】MC_102_SafetyDrive_DataModel.py

```
状态价值函数计算结果(数组) : [ 1.03  1.72  2.72  3.02 -5.17 -6.73  6.   -2.37 -1.    5.    0.  ]
Start:       1.03
Normal:      1.72
Pedestrians: 2.72
DownSpeed:   3.02
ExceedSpeed:-5.17
RedLight:   -6.73
LowSpeed:    6.0
MobilePhone:-2.37
Crash:      -1.0
Goal:        5.0
End:         0.0
```

#### 性能衡量算法



### 10.2.2 每次访问法

#### 算法描述

【算法 10.2.1】每次访问型蒙特卡洛法。

下面的伪代码中，$\leftarrow$ 表示赋值，$\Leftarrow$ 表示追加列表。

---

输入：起始状态$s,\gamma$, Episodes
初始化：$G(S) \leftarrow 0, N(S) \leftarrow 0$
多幕 Episodes 循环：
　　列表置空 $T = [\ ] $ 用于存储序列数据 $(s,r)$
　　幕内循环直到终止状态：
　　　　从 $s$ 根据环境模型得到 $s',r$ 以及是否终止的标志
　　　　$T \Leftarrow (s',r)$
　　　　$s \leftarrow s'$
　　　　$G_t \leftarrow 0$
　　对 $T$ 从后向前遍历, $t=\tau-1,\tau-2,...,0$
　　　　从 $T$ 中取出 $s_t,r_t$
　　　　$G_t \leftarrow \gamma G_t+r_t$
　　　　$G(s_t) \leftarrow G(s_t)+G_t$
　　　　$N(s_t) \leftarrow N(s_t)+1$
$V(S) \leftarrow G(S) / N(S)$
输出：$V(S)$

---

为什么叫做“每次访问法”呢？因为在算法中，每次只要遇到 $s_t$ 就要计算它的 $G$ 值。

#### 算法说明

<center>
<img src="./img/MC-2.png">

图 10.2.2 算法说明
</center>

其中，状态下标 $S,N,L,G,E$ 等是对应的状态的缩写，无关紧要，但它们是一个真实的序列。

#### 算法实现

```python
# MC2 - 反向计算G值，记录每个状态的G值，每次访问型
def MC_EveryVisit(dataModel, start_state, episodes, gamma):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Ts = []     # 一幕内的状态序列
        Tr = []     # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):            # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Ts.append(s.value)
            Tr.append(r)
            s = next_s

        num_step = len(Ts)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = Ts[t]
            r = Tr[t]
            G = gamma * G + r
            Value[s] += G     # 值累加
            Count[s] += 1     # 数量加 1
    
    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    return Value / Count    # 求均值
```

在此我们先不验证该算法的的性能，等到讲解完下个算法后一起验证并比较。

### 10.2.3 首次访问法

既然有每次访问法，当然就会有首次访问法。

比如在一个采样序列中，有如下状态数据：$s_a, s_b, s_c, s_a, s_e, s_b$，首次访问法在倒序遍历到最后那个 $s_b$ 时，先检查 $s_b$ 在前面时候出现过，在本例中出现过，所以就不会在这个 $t$ 上计算 $s_b$ 的 $G_t$ 值。在这个例子中，最终会计算的是 $s_a, s_b, s_c, s_e$ 四个状态的 $G$ 值。

马尔科夫链的环状结构会使得这种情况非常常见，显然第一个 $s_a$ 和 第四个 $s_a$ 在时域上 $t$ 应该具有不同的价值，但是从静态角度看，它们又应该具有相同的价值。Sutton 在研究了这个问题后，认为两者具有不同的理论基础，根据大数定律，首次访问法的方差以 $1/\sqrt{n}$ 的速度收敛，每次访问法也会渐近收敛到真实的状态价值。


#### 算法描述

【算法 10.2.1】首次访问型蒙特卡洛法。

下面的伪代码中，$\leftarrow$ 表示赋值，$\Leftarrow$ 表示追加列表。

---

输入：起始状态$s,\gamma$, Episodes
初始化：$G(S) \leftarrow 0, N(S) \leftarrow 0$
多幕 Episodes 循环：
　　列表置空 $T = [\ ] $ 用于存储序列数据 $(s,r)$
　　幕内循环直到终止状态：
　　　　从 $s$ 根据环境模型得到 $s',r$ 以及是否终止的标志
　　　　$T \Leftarrow (s',r)$
　　　　$s \leftarrow s'$
　　　　$G_t \leftarrow 0$
　　对 $T$ 从后向前遍历, $t=\tau-1,\tau-2,...,0$
　　　　从 $T$ 中取出 $s_t,r_t$
　　　　如果 $s_t$ 不在 $s_0,s_1,\cdots,s_{t-1}$ 中：
　　　　　　$G_t \leftarrow \gamma G_t+r_t$
　　　　　　$G(s_t) \leftarrow G(s_t)+G_t$
　　　　　　$N(s_t) \leftarrow N(s_t)+1$
$V(S) \leftarrow G(S) / N(S)$
输出：$V(S)$

---






#### 算法实现

```Python
# MC2 - 反向计算G值，记录每个状态的G值，首次访问型
def MC_FirstVisit(dataModel, start_state, episodes, gamma):
    Value = np.zeros(dataModel.nS)  # G 的总和
    Count = np.zeros(dataModel.nS)  # G 的数量
    for episode in tqdm.trange(episodes):   # 多幕循环
        Ts = []     # 一幕内的状态序列
        Tr = []     # 一幕内的奖励序列
        s = start_state
        is_end = False
        while (is_end is False):            # 幕内循环
            next_s, r, is_end = dataModel.step(s)   # 从环境获得下一个状态和奖励
            Ts.append(s.value)
            Tr.append(r)
            s = next_s

        num_step = len(Ts)
        G = 0
        # 从后向前遍历计算 G 值
        for t in range(num_step-1, -1, -1):
            s = Ts[t]
            r = Tr[t]
            G = gamma * G + r
            if not (s in Ts[0:t]):# 首次访问型（坚持该状态在前面的序列中有无出现）
                Value[s] += G     # 值累加
                Count[s] += 1     # 数量加 1
    
    Count[Count==0] = 1 # 把分母为0的填成1，主要是终止状态
    return Value / Count    # 求均值
```



<center>
<img src="./img/MC-2-RMSE.png">

图 2
</center>

### 运行结果比较

表   $\gamma=1$ 时的状态值计算结果比较

|状态|矩阵法|顺序计算法|首次访问法|每次访问法|
|-|-:|-:|-:|-:|
|出发 Start|           1.03|1.11|1.03|1.03|
|正常行驶 Normal|      1.72|1.52|1.71|1.7|
|礼让行人 Pedestrians| 2.72|2.70|2.73|2.71|
|闹市减速 DownSpeed|   3.02|2.79|2.83|2.83|
|超速行驶 ExceedSpeed| -5.17|-5.19|-5.2|-5.2|
|路口闯灯 RedLight|    -6.73|-6.71|-6.71|-6.71|
|小区减速 LowSpeed|     6.00|6.00|6.00|6.00|
|拨打电话 MobilePhone| -2.37|-2.51|-2.28|-2.3|
|发生事故 Crash|       -1.00|-1.00|-1.00|-1.00|
|安全抵达 Goal|        5.00|5.00|5.00|5.00|
|终止 End|              0.00| 0.00|0.00|0.00|
|**误差 RMSE**|-|**0.102**|**0.063**|**0.061**|
|每状态重复次数|-|**5000**|**2000**|**2000**|
|耗时|-|**7.37**|**4.70**|**4.24**|


可以看到，改进的算法在速度上和精度上都比原始算法要好。最后一行不是状态值，是 RMSE 的误差值，原始算法误差为 0.042，改进算法为 0.034，越小越好。

从性能上看，原始算法对每个状态做了 10000 次采样，相当于一共 $11 \times 10000=110000$ 次采样。改进算法对所有状态（混合）一共做了 50000 次采样。



### 参考资料

https://new.qq.com/omn/20220314/20220314A09IY600.html