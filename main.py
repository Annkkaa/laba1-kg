import numpy as np # библиотеки
from PIL import Image
from math import sqrt

W, H = 1000, 1000 # размеры окна
img_mnt = np.zeros((H, W, 3), dtype=np.uint8) # буфер изображения (черное)
z_buffer = np.full((H, W), 1e9, dtype=np.float64) # z-буфер

v = [] # вершины
f = [] # грани

with open('model.obj', 'r') as file: # открываем файл в режимы чтения
    for s in file:
        sp = s.split() # разбиваем строку
        if not sp: # пропускаем пустые строки
            continue
        if sp[0] == 'v': # если вершина
            v.append([float(sp[1]), float(sp[2]), float(sp[3])]) # добавляем
        elif sp[0] == 'f': # если грань
            face = []
            for sf in sp[1:4]: # берем вершины треугольника
                sf_parts = sf.split('/') # разбиваем
                sf_index = int(sf_parts[0])
                face.append(sf_index) # добавляем индекс в грань
            f.append(face) # добавляем грань

def draw_triangle_numpy(image, z_buffer, v0, v1, v2, z0, z1, z2, color=(255, 255, 255)): # функция отрисовки треугольника
    H_img, W_img = image.shape[:2] # размеры
    x0, y0 = v0 # распаковываем координаты вершин
    x1, y1 = v1
    x2, y2 = v2

    ix_min = max(0, int(np.floor(min(x0, x1, x2)))) # левая граница
    ix_max = min(W_img - 1, int(np.ceil(max(x0, x1, x2)))) # правая граница
    iy_min = max(0, int(np.floor(min(y0, y1, y2)))) # верхняя граница
    iy_max = min(H_img - 1, int(np.ceil(max(y0, y1, y2)))) # нижняя граница

    if ix_min > ix_max or iy_min > iy_max: # если вне экрана выходим
        return

    xs = np.arange(ix_min, ix_max + 1) # массивы координат x,y
    ys = np.arange(iy_min, iy_max + 1)
    X, Y = np.meshgrid(xs, ys) # сетка координат

    denom = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2) # знаменатель барицентрических координат
    if denom == 0.0: # если треугольник вырожденный выходим
        return

    l0 = ((X - x2) * (y1 - y2) - (x1 - x2) * (Y - y2)) / denom # l0 координата
    l1 = ((x0 - x2) * (Y - y2) - (X - x2) * (y0 - y2)) / denom # l1 координата
    l2 = 1.0 - l0 - l1  # l2 координата

    mask = (l0 >= 0.0) & (l1 >= 0.0) & (l2 >= 0.0) # маска пикселей внутри треугольника

    z = l0 * z0 + l1 * z1 + l2 * z2 # интерполяция глубины треугольника

    z_slice = z_buffer[iy_min:iy_max + 1, ix_min:ix_max + 1] # срез z-буфера под наши массивы координат
    z_mask = mask & (z < z_slice) # рисуем только если он ближе

    image[iy_min:iy_max + 1, ix_min:ix_max + 1][z_mask] = color # закрашиваем только то что нужно
    z_buffer[iy_min:iy_max + 1, ix_min:ix_max + 1][z_mask] = z[z_mask] # обновляем глубину

for i in range(len(f)): # проходимся по граням
    p0 = v[f[i][0] - 1] # вершины
    p1 = v[f[i][1] - 1]
    p2 = v[f[i][2] - 1]

    ux = p1[0] - p2[0] # векторы u = p1 - p2
    uy = p1[1] - p2[1]
    uz = p1[2] - p2[2]

    vx = p1[0] - p0[0] # векторы v = p1 - p0
    vy = p1[1] - p0[1]
    vz = p1[2] - p0[2]

    nx = uy * vz - uz * vy # нормали u * v
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx

    norm_n = sqrt(nx*nx + ny*ny + nz*nz) # длина нормали

    if norm_n == 0: # пропускаем если вырожденный треугольник
        continue

    cos_alpha = nz / norm_n # освещенность

    if cos_alpha < 0: # только если смотрит в камеру
        intensity = int(-255 * cos_alpha) # вычисляем освещенность
        intensity = max(0, min(255, intensity)) # ограничеваем от 0 до 255
        color = (intensity, 0, 0) # красный с освещенностью

        x0 = 10000 * p0[0] + 500 # координаты с учетом масштаба и смещения (для красивого отображения)
        y0 = -10000 * p0[1] + 1000
        x1 = 10000 * p1[0] + 500
        y1 = -10000 * p1[1] + 1000
        x2 = 10000 * p2[0] + 500
        y2 = -10000 * p2[1] + 1000

        z0 = p0[2] # глубины вершин берем из z координат
        z1 = p1[2]
        z2 = p2[2]

        draw_triangle_numpy(img_mnt, z_buffer, (x0, y0), (x1, y1), (x2, y2), z0, z1, z2, color=color) # рисуем

img = Image.fromarray(img_mnt) # превращаем в изображение
img.save('img.png') # сохраняем