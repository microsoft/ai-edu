import random

def host_open_door(doors, gift_door, selected_door):
    host_open_candidate = doors.copy()
    # 主持人不能打开奖品所在的门, 也不能打开客户选择的门
    remove_doors = [gift_door, selected_door]
    for door in doors:
        if (door in remove_doors):
            host_open_candidate.remove(door)
    # print(doors)
    # 主持人随机打开一个没有奖品的门
    opened_door = random.choice(host_open_candidate)
    return opened_door

def player_change_door(doors, selected_door, opened_door):
    player_change_candidate = doors.copy()
    # 参赛者不能选择持人打开的门, 也不能打开自己第一次选择的门
    remove_doors = [opened_door, selected_door]
    for door in doors:
        if (door in remove_doors):
            player_change_candidate.remove(door)
    # print(doors)
    # 参赛者随机打开一个剩余的门
    changed_door = random.choice(player_change_candidate)
    return changed_door

# n_doors: 一共有几扇门（>=3)
# b_change: 是否换门，yes=换
# return: True=中奖
def try_once(n_doors: int):
    # 一共3扇门
    doors = [i for i in range(n_doors)]

    # 汽车奖品随机在n扇门后，忽略山羊（那只是个讽刺）
    gift_door = random.randint(0, n_doors-1)
    # print("gift_door=", gift_door)

    # 客户从n扇门中随机选一个
    selected_door = random.randint(0, n_doors-1)
    # print("selected_door=", selected_door)

    # 主持人打开一扇门
    opened_door = host_open_door(doors, gift_door, selected_door)
    # 参赛者更换了一扇门
    changed_door = player_change_door(doors, selected_door, opened_door)
    # 如果参赛者不更换门而中奖
    no_change_win = 1 if (gift_door == selected_door) else 0
    # 参赛者更换门而中奖
    win_after_change = 1 if (gift_door == changed_door) else 0

    return no_change_win, win_after_change

if __name__ == "__main__":
    total = 100000
    n_doors = 8
    n_win_after_change = 0
    n_no_change_win = 0
    for i in range(total):
        #print(i)
        no_change_win, win_after_change = try_once(n_doors)
        n_win_after_change += win_after_change
        n_no_change_win += no_change_win

    print(n_win_after_change, n_no_change_win)
    print(str.format("更换选择而中奖的概率={0} \n不更换而中奖的概率={1}", 
                     n_win_after_change/total, n_no_change_win/total))
