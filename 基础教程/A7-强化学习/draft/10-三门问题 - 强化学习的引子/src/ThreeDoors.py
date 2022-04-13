import random

# 随机选择一扇不在forbidden_doors列表中存在的门
# 该列表可以是空，表示可以在所有doors中选择
def choice_one_door(doors, forbidden_doors):
    choice_candidate = doors.copy()
    for door in doors:
        if (door in forbidden_doors):
            choice_candidate.remove(door)
    # print(doors)
    changed_door = random.choice(choice_candidate)
    return changed_door

# n_doors: 一共有几扇门（>=3)
# return: 如果不换门而中奖，返回 (1,0)
#         如果更换门而中奖，返回（0,1)
def try_once(n_doors: int):
    # 一共n扇门
    doors = [i for i in range(n_doors)]
    # 汽车随机在n扇门之一后，忽略山羊（那只是个讽刺）
    gift_door = choice_one_door(doors, [])
    # 参赛者从n扇门中随机选一个
    first_choice = choice_one_door(doors, [])
    # 主持人打开一扇门（除了参赛者的首选门和汽车所在的门以外）
    opened_door = choice_one_door(doors, [gift_door, first_choice])
    # 参赛者更换了一扇门（不能选自己的首选门和主持人打开的门）
    second_choice = choice_one_door(doors, [first_choice, opened_door])
    # 如果参赛者不更换门而中奖
    no_change_but_win = 1 if (gift_door == first_choice) else 0
    # 参赛者更换门而中奖
    win_after_change = 1 if (gift_door == second_choice) else 0

    return no_change_but_win, win_after_change


def try_n_doors(n_doors):
    total = 100000
    n_win_0 = 0
    n_win_1 = 0
    for i in range(total):
        no_change_but_win, win_after_change = try_once(n_doors)
        n_win_0 += win_after_change
        n_win_1 += no_change_but_win
    print(str.format("{0}扇门:", n_doors))
    print(n_win_0, n_win_1)
    print(str.format("更换选择而中奖的概率={0} \n不换选择而中奖的概率={1}", 
                     n_win_0/total, n_win_1/total))

if __name__ == "__main__":
    n_doors = 3
    try_n_doors(n_doors)

    n_doors = 8
    try_n_doors(n_doors)
