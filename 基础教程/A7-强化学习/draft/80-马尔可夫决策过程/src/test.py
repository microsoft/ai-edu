
import numpy as np

s_id = 0
Pos2Sid = {}
Sid2Pos = {}
for y in range(3):
    for x in range(4):
        Pos2Sid[x,y] = s_id
        Sid2Pos[s_id] = [x,y]
        s_id += 1

print(Pos2Sid)
print(Sid2Pos)

for s,(x,y) in Sid2Pos.items():
    print(s)
    print(x,y)