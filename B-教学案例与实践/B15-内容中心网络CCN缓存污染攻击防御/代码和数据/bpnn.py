from os import path,listdir,getcwd,system
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

attacktime=[31,91,151,201,251,301,401,501,601,701,801,901,1001,1501,1601,1701,1801,1901,2001,2151,2201,2401,2501,2701,2801,2901,2651,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,751,752,753,754,755,756,757,758,759,760,761,851,852,853,854,855,856,857,858,859,860,861,3101,3201,3351,3401,3501,3601,3651,3701,3801,3901,4001,4101,4201,4301,4401,4501,4601,4701,4801,4901,5001,5101,5231,5270,5378,5469,5556,5685,5779,5851,5901,5902,5903,5904,5905,5906,5907,5908,5909,5910,6001,6101,6201,6351,6401,6501,6601,6701,6801,7001,7101,7201,7302,7401,7501,7601,7701,7801,7891,8811,8951,8961,9001,9101,9201,9301,9451,9513,9601,9701,9801,9898,9951,9952,9953,9954,9955,9956,9957,9958]

# 处理输入数据
def process(fname='cs-trace.txt'):
    ret=[]    # ret存储数据 
    ret2=[]    # cnt存储数据对应的标签 
    if not path.exists('data.txt'):
        fp=open(fname)
        cnt=0  # cnt记录当前处理的行号  
        tmp=[] # tmp用于存储每一行的数据
        val=0  # val用于暂存读取的数据  
        for lines in fp.readlines(): # 读取数据  
            wd=lines.split()
            if wd[0]=='Time':
                continue
            cnt+=1
            if cnt%2==1:        # 读取在奇数行的缓存命中次数 
                val=int(wd[-1]) 
            else:               # 读取在偶数行的缓存未命中次数并计算命中率  
                if int(wd[-1])+val==0:  
                    val=1           # 如果Hit和Miss都为0，设置命中率100%  
                else:
                    val=val/(int(wd[-1])+val) # 计算缓存命中率
                tmp.append(val)               # 添加一个节点的缓存命中率
            if cnt==50:                       # 开始添加标签  
                cnt=0
                if int(wd[0]) in attacktime:
                    ret2.append([1]) # 如果是攻击时刻则把标签设为1
                else:
                    ret2.append([0]) # 如果不是攻击时刻则把标签设为0
                ret.append(tmp) # 添加一行的数据
                tmp=[]
        fp.close()
    return ret,ret2 # 返回数据和对应的标签

# 计算成功判断攻击发生的正确率
def calAccuracy(preRes,label):
    cnt=0
    tot=0
    acc=0
    for i in preRes:
        # 预测概率大于0.9且标签为1判定位预测正确
        if i>=0.9 and label[cnt]==1:
            tot+=1
            acc+=1
        elif label[cnt]==1:
            tot+=1
        cnt+=1
    return acc/tot # 返回正确率

if __name__=='__main__':
    ret,ret2=process()

    # 设置训练集为7000行数据，测试集为3000行数据。
    x_train=np.array(ret[:7000])
    y_train=np.array(ret2[:7000])
    x_test=np.array(ret[3000:])
    y_test=np.array(ret2[3000:])

    # 设置输入数据的维度为25
    inputs=tf.keras.Input(shape=(25,))
    
    # 添加一个全连接层，激活函数选用ReLU
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    
    # 输出层维度位1,激活函数选用sigmoid
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)

    # 设置损失函数为对数损失，优化函数使用rmsprop
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    # 训练模型，迭代次数30次，batch size设置成100
    model.fit(x_train, y_train,
            epochs=30,
            batch_size=100)
    print('-----------------------------------')
    # 对测试集进行预测
    ans=model.predict(x_train)
    print("The result is %f"%(calAccuracy(ans,y_train)))