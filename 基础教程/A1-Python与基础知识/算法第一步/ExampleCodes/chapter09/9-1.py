tn = 165  # 这里可以是任意整数
found = False

for i in range(1,1001):
    if i == tn:
        print("secrete number is ", i)
        found = True
        break

if not found:
    print("failed")