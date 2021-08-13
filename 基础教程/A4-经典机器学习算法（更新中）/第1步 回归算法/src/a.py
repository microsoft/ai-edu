def main():
    s , res= ['a','b','c', 'd', 'e'] , []
    l = len(s)
    for i in range(l):
        for j in range(l):
            if i == 0:
                res.append(''.join(s[j]))
            elif j+1 <= l :
                for x in range(j+1,l):
                    if x+i > l : break
                    ss = str(s[j])+''.join(s[x:x+i])
                    res.append(ss)
               
    print(res,len(res))



def abc(input):
    count = len(input)
    result = []
    result.append(input[0])
    for i in range(1,count):
        # copy the existing units
        result_backup = result[:]
        # append new char (input[i]) to all the existing result
        # e.g. a,b,ab->ac,bc,abc
        for j in range(len(result)):
            result[j] = result[j] + input[i]
        # a,b,ab "+" ac,bc,abc
        result.extend(result_backup)
        # append 'c'
        result.append(input[i])
    return result

def abcd(input):
    count = len(input)
    result = []
    for i in range(0,count):
        # copy the existing units
        result_backup = result[:]
        # append new char (input[i]) to all the existing result
        # e.g. a,b,ab->ac,bc,abc
        for j in range(len(result)):
            result[j] = result[j] + input[i]
        # a,b,ab "+" ac,bc,abc
        result.extend(result_backup)
        # append 'c'
        result.append(input[i])
    return result


if __name__ == '__main__':
    r = abcd(['a', 'b', 'c', 'd'])
    print(len(r), sorted(r))
    