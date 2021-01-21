from SearchAlgorithms import binary_search

filePath = 'sequences_for_search.txt'

with open(filePath) as fp:
    line = fp.readline()
    while line:
        tmps = line.strip().split('|')  # 将读入的一行用“|”分隔为两段

        if len(tmps) != 2:  # 如果格式不对，则忽略此行
           continue

        arr_strs = tmps[0].strip().split(',')  # 将“|”前的部分再以逗号进行分割
        arr = list(map(int, arr_strs))  # 转化为整型List
        tn = int(tmps[1])  # 将“|”后的部分读取为整型

        result = binary_search(arr, tn)

        if result >= 0:
            print("Succeeded! The target index is: ", result)
        else:
            print("Search failed.")
        line = fp.readline()