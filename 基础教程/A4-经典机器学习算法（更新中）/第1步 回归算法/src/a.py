from itertools import permutations
s = 'abc'
for i in range(1, len(s)+1):
    for result in permutations(s, i):
        print(result)
