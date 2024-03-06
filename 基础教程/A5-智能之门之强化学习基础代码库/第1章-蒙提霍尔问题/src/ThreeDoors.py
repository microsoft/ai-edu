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
    win_without_change = 1 if (gift_door == first_choice) else 0
    # 参赛者更换门而中奖
    win_after_change = 1 if (gift_door == second_choice) else 0

    return win_without_change, win_after_change


def try_n_doors(n_doors):
    total = 100000
    win_without_change_all = 0
    win_after_change_all = 0
    for i in range(total):
        win_without_change, win_after_change = try_once(n_doors)
        win_without_change_all += win_without_change
        win_after_change_all += win_after_change
    print(str.format("{0}扇门:", n_doors))
    print(str.format("更换选择而中奖的次数={0}, 概率={1}", win_after_change_all, win_after_change_all/total))
    print(str.format("不换选择而中奖的次数={0}, 概率={1}", win_without_change_all, win_without_change_all/total))

if __name__ == "__main__":
    n_doors = 3
    try_n_doors(n_doors)

    n_doors = 8
    try_n_doors(n_doors)
