import numpy as np
from PIL import Image
from math import floor
img_mnt = np.zeros((1000,1000,3), dtype=np.uint8)

file = open('model.obj')
v = []
f = []

def line(img_mat,x0,y0,x1,y1):
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    dmax = max(dx,dy)
    L = int(dmax) + 1

    if L == 1:
        px = floor(x0)
        py = floor(y0)
        if 0 <= px < 1000 and 0 <= py < 1000:
            img_mat[py,px] = 255
        return
    
    steps = L-1
    dx = (x1-x0) / steps
    dy = (y1-y0) / steps
    x,y = x0,y0

    for _ in range(L):
        px = floor(x)
        py = floor(y)

        if 0 <= px < 1000 and 0 <= py < 1000:
            img_mat[py,px] = 255
        x += dx
        y += dy

for s in file:
    sp = s.split()
    if (sp[0] == 'v'):
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])
    if(sp[0] == 'f'):
        face = []
        for sf in sp[1:4]:
            sf_parts = sf.split('/')
            sf_index = int(sf_parts[0])
            face.append(sf_index)
        f.append(face)

print(f)

for i in range(len(f)):
    x0 = float(10000* v[f[i][0]-1][0]+500)
    y0 = float(-10000* v[f[i][0]-1][1]+1000)
    x1 = float(10000* v[f[i][1]-1][0]+500)
    y1 = float(-10000* v[f[i][1]-1][1]+1000)
    x2 = float(10000* v[f[i][2]-1][0]+500)
    y2 = float(-10000* v[f[i][2]-1][1]+1000)

    line(img_mnt,x0,y0,x1,y1)
    line(img_mnt,x1,y1,x2,y2)
    line(img_mnt,x2,y2,x0,y0)

img = Image.fromarray(img_mnt)
img.save('img.png')