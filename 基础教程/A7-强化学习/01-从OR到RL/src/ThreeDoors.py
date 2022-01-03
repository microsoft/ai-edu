import random

def try_once():
    # 一共3扇门
    doors = [0, 1, 2]

    # 奖品随机在3扇门后
    gift_door = random.randint(0, 2)
    # print("gift_door=", gift_door)

    # 客户从3扇门中随机选一个
    customer_door = random.randint(0,2)
    # print("customer_door=", customer_door)
    
    # 主持人不能打开奖品所在的门
    doors.remove(gift_door)
    # print(doors)

    # 主持人不能打开客户选择的门
    if (customer_door in doors):
        doors.remove(customer_door)
    # print(doors)

    if (gift_door == customer_door):
        return True
    else:
        return False

if __name__ == "__main__":
    win = 0
    loss = 0
    total = 1000000
    for i in range(total):
        #print(i)
        if (try_once() is True):
            win += 1
        else:
            loss += 1
    print(str.format("win ratio={0}, loss ratio={1}", 
                     win/total, loss/total))
