low = 1
high = 1000

while low <= high:
    m = int((high - low) / 2) + low
    print("My guess is", m)

    # user_input是循环条件中被判断的变量，因此需要在循环之前先有一个值，否则循环会出错
    user_input = ""
    while user_input != '1' and user_input != '2' and user_input != '3':

        print("\t\t1) Bingo! %s is the secret number! \n\
        2) %s < the secret number.\n\
        3) %s > the secret number." % (m, m, m))

        user_input = input("Your option:")
        user_input = user_input.strip()

    if user_input == '1':
        print("Succeeded! The secret number is %s." % m )
        break
    else:
        if user_input == '2':
            low = m + 1
        else:
            high = m - 1

if low > high:
    print("Failed！")