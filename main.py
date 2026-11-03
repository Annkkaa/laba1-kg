import numpy as np
from PIL import Image, ImageTk
from math import sqrt
import tkinter as tk
from numba import njit

WINDOW_W = 1000 # ширина картинки
WINDOW_H = 1000 # высота картинки
Z_BUFFER_INIT = 1e9 # инициализация z buffer (максимально далеко)
COLOR_MAX = 255 # максимальная яркость
COLOR_MIN = 0 # минимальная яркость
INTENSITY_SCALE = -255.0 # коэф для яркости
PROJ_SCALE = 10000 # коэф для увеличения модели
PROJ_SHIFT_X = 500 # коэф для сдвига по х
PROJ_SHIFT_Y = 1000 # коэф для сдвига по y
OBJ_FILENAME = 'model.obj' # название модели
OUTPUT_FILENAME = 'img.png' # название картинки
APP_WIN_W = 900 # ширина приложения
APP_WIN_H = 900 # высота приложения

class Scene:
    def __init__(self, filename): # конструктор
        self.vertices = [] # создание списка вершин
        self.faces = [] # создание списка граней
        self.load_obj(filename) # обработка модели в списки

    def load_obj(self, filename): # функция обработки модели
        with open(filename, 'r') as file: # открываем файл в режиме чтения
            for s in file:
                sp = s.split() # разбиваем строку
                if not sp: # пропускаем пустые строки
                    continue
                if sp[0] == 'v': # если вершина
                    self.vertices.append([float(sp[1]), float(sp[2]), float(sp[3])]) # добавляем
                elif sp[0] == 'f': # если грань
                    face = []
                    for sf in sp[1:4]: # берем вершины треугольника
                        sf_parts = sf.split('/') # разбиваем
                        sf_index = int(sf_parts[0]) - 1 # индекс вершины (с 0)
                        face.append(sf_index) # добавляем индекс в грань
                    self.faces.append(face) # добавляем грань

@njit(cache=True)
def rasterize_triangle(img, zbuf, px0, py0, px1, py1, px2, py2, z0, z1, z2, cr, cg, cb): # декоратор
    H_img = img.shape[0] # высота изображения
    W_img = img.shape[1] # ширина изображения

    ix_min_f = min(min(px0, px1), px2) # минимальные и максимальные X и Y среди трех вершин
    ix_max_f = max(max(px0, px1), px2)
    iy_min_f = min(min(py0, py1), py2)
    iy_max_f = max(max(py0, py1), py2)

    ix_min = max(0, int(ix_min_f)) # минимальные и максимальные X и Y но целые
    ix_max = min(W_img - 1, int(ix_max_f))
    iy_min = max(0, int(iy_min_f))
    iy_max = min(H_img - 1, int(iy_max_f))

    if ix_min > ix_max or iy_min > iy_max: # невалидный треугольник
        return 0

    denom = (px0 - px2) * (py1 - py2) - (px1 - px2) * (py0 - py2) # знаменатель барицентрических координат
    if denom == 0.0: # треугольник вырожден
        return 0

    inv_denom = 1.0 / denom # обратное значение для умножения

    for py in range(iy_min, iy_max + 1): # перебираем каждый пиксель
        for px in range(ix_min, ix_max + 1):
            l0 = ((px - px2) * (py1 - py2) - (px1 - px2) * (py - py2)) * inv_denom # барицентрические координаты
            l1 = ((px0 - px2) * (py - py2) - (px - px2) * (py0 - py2)) * inv_denom
            l2 = 1.0 - l0 - l1

            if l0 >= 0.0 and l1 >= 0.0 and l2 >= 0.0: # пиксель находится внутри треугольника
                z = l0 * z0 + l1 * z1 + l2 * z2 # вычисление глубины
                if z < zbuf[py, px]: # проверка на глубину ближе к камере чем в буфере
                    zbuf[py, px] = z # обновление z буфер
                    img[py, px, 0] = cr # обновление цветов RGB
                    img[py, px, 1] = cg
                    img[py, px, 2] = cb

    return 1 # успешно нарисовано


@njit(cache=True)
def render_mesh(img, zbuf, vertices, faces, proj_scale, shift_x, shift_y): # отрисовка сетки
    num_faces = faces.shape[0] # число граней
    drawn = 0 # количество нарисованных

    for i in range(num_faces): # пробегаемся по граням
        fi0 = faces[i, 0] # получение индексов трех вершин из треугольника
        fi1 = faces[i, 1]
        fi2 = faces[i, 2]

        x0 = vertices[fi0, 0] # извлечение 3D координат
        y0 = vertices[fi0, 1]
        z0 = vertices[fi0, 2]
        x1 = vertices[fi1, 0]
        y1 = vertices[fi1, 1]
        z1 = vertices[fi1, 2]
        x2 = vertices[fi2, 0]
        y2 = vertices[fi2, 1]
        z2 = vertices[fi2, 2]

        ux = x1 - x2 # вектор u
        uy = y1 - y2
        uz = z1 - z2

        vx = x1 - x0 # вектор v
        vy = y1 - y0
        vz = z1 - z0

        nx = uy * vz - uz * vy # вектор нормали n
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx

        norm_n = sqrt(nx * nx + ny * ny + nz * nz) # вычисление длины нормали

        if norm_n == 0.0: # если равно нулю то пропускаем                          
            continue

        cos_alpha = nz / norm_n # нормализуем нормаль и берем ее z                                                              

        if cos_alpha < 0.0:
            intensity = int(INTENSITY_SCALE * cos_alpha)
            if intensity < COLOR_MIN:
                intensity = COLOR_MIN
            if intensity > COLOR_MAX:
                intensity = COLOR_MAX

            px0 = int(proj_scale * x0 + shift_x)
            py0 = int(-proj_scale * y0 + shift_y)
            px1 = int(proj_scale * x1 + shift_x)
            py1 = int(-proj_scale * y1 + shift_y)
            px2 = int(proj_scale * x2 + shift_x)
            py2 = int(-proj_scale * y2 + shift_y)

            rasterize_triangle(img, zbuf, px0, py0, px1, py1, px2, py2, z0, z1, z2, intensity, 0, 0)
            drawn += 1

    return drawn

class Renderer:
    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.img = np.zeros((height, width, 3), dtype=np.uint8)
        self.zbuf = np.full((height, width), Z_BUFFER_INIT, dtype=np.float64)
        self.proj_scale = PROJ_SCALE
        self.shift_x = PROJ_SHIFT_X
        self.shift_y = PROJ_SHIFT_Y

    def render(self, scene):
        verts = np.array(scene.vertices, dtype=np.float64)
        faces_arr = np.array(scene.faces, dtype=np.int64)
        render_mesh(self.img, self.zbuf, verts, faces_arr, self.proj_scale, self.shift_x, self.shift_y)

    def save(self, filename):
        img = Image.fromarray(self.img)
        img.save(filename)

class App:
    def __init__(self, scene, renderer, win_w=APP_WIN_W, win_h=APP_WIN_H):
        self.scene = scene
        self.renderer = renderer

        self.root = tk.Tk()
        self.root.geometry(f"{win_w}x{win_h}")
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.focus_set()
        self.canvas.bind('<Configure>', lambda e: self._refit())
        self.root.bind('<Key>', self.on_key)
        self.photo = None
        self.last_full = None
        self.render()

    def on_key(self, event):
        k = event.keysym
        if k == 'space':
            Image.fromarray(self.renderer.img).save(OUTPUT_FILENAME)
            return 'break'
        else:
            return
        self.render()
        return 'break'

    def render(self):
        self.renderer.render(self.scene)
        self.last_full = Image.fromarray(self.renderer.img)
        self._refit()

    def _refit(self):
        if self.last_full is None:
            return
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        if ww < 10 or wh < 10:
            return
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.last_full.resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)

    def run(self):
        self.root.mainloop()

def main():
    scene = Scene(OBJ_FILENAME)

    renderer = Renderer(WINDOW_W, WINDOW_H)

    app = App(scene, renderer)
    app.run()


if __name__ == '__main__':
    main()