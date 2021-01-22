def test_scalar_param(a):
    a = a * 2
    return a


x = 3
y = test_scalar_param(x)

print("x is", x)
print("y is", y)