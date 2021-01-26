low = 1
high = 1000

loop_num = 0  # 记录循环轮数

while low <= high:
    m = int((high - low) / 2) + low
    print("My guess is", m)

    # userInput是循环条件中被判断的变量，因此需要在循环之前先有个值，否则循环会出错
    user_input = ""
    input_num = 0

    while user_input != '1' and user_input != '2' and user_input != '3':
        if input_num == 3:
            print("\nYou input too many invalid options. Game over!")
            exit(0)

        print("\t\t1) Bingo! %s is the secret number! \n\
        2) %s < the secret number.\n\
        3) %s > the secret number." % (m, m, m))

        user_input = input("Your option:")
        user_input = user_input.strip()

        input_num += 1

    input_num = 0
    loop_num += 1

    if user_input == '1':
        print("Succeeded! The secret number is %s.\n\
        It took %s round to locate the secret number. \n" % (m, loop_num))
        break
    else:
        if user_input == '2':
            low = m + 1
        else:
            high = m - 1

if low > high:
    print("Failed！Cannot got your secret number. Make sure it in range of [1, 1000].")