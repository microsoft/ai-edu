low = 1
high = 1000
loop_num = 0

while low <= high:
    m = int((high - low) / 2) + low

    loop_num += 1
    print("[Loop %s]: My guess is %s, low is %s, and high is %s" % (loop_num, m, low, high))
    user_input = ""

    while user_input != '1' and user_input != '2' and user_input != '3' :
        print("\t\t1) %s == sn \n\
        2) %s < sn.\n\
        3) %s > sn." % (m, m, m))

        user_input = input("Your option:")
        user_input = user_input.strip()

    if user_input == '1':
        print("Succeeded! SN is: ", m)
        break
    else:
        if user_input == '2':
            low = m + 1
        else:
            high = m - 1

if low > high:
    print("Failed！Cannot got your secret number. Make sure it in range of [1, 1000].")