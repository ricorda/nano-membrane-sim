"""viz.py —— 共用的小画板：坐标、刻度、热图、连线。

只用 PIL。风格和 sim.py 里的图一致（同一套颜色、同一套字体）。

为什么单独一个文件
------------------
    scan/gp/al 三个脚本都要画，复制三遍容易长歪。
"""

import os

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"

INK = (28, 30, 34)
GREY = (120, 124, 132)
LIGHT = (186, 192, 202)
BLUE = (44, 92, 178)
PALE = (158, 184, 224)
RED = (196, 72, 60)
GREEN = (32, 122, 88)
AMBER = (196, 138, 40)
BG = (255, 255, 255)


def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


class Axes:
    """一块绘图区：把数据坐标映射到像素，顺便画框和刻度。"""

    def __init__(self, dr, box, xlim, ylim):
        self.dr = dr
        self.x0, self.y0, self.x1, self.y1 = box      # 左上、右下
        self.xlim = xlim
        self.ylim = ylim

    def px(self, x):
        a, b = self.xlim
        return self.x0 + (x - a) / (b - a) * (self.x1 - self.x0)

    def py(self, y):
        a, b = self.ylim
        return self.y1 - (y - a) / (b - a) * (self.y1 - self.y0)

    def frame(self):
        self.dr.rectangle([self.x0, self.y0, self.x1, self.y1], outline=LIGHT, width=1)

    def xticks(self, vals, fmt="{:g}", size=14):
        for v in vals:
            x = self.px(v)
            self.dr.line([(x, self.y1), (x, self.y1 + 5)], fill=LIGHT, width=1)
            self.dr.text((x, self.y1 + 10), fmt.format(v), font=font(size),
                         fill=GREY, anchor="mt")

    def yticks(self, vals, fmt="{:g}", size=14):
        for v in vals:
            y = self.py(v)
            self.dr.line([(self.x0 - 5, y), (self.x0, y)], fill=LIGHT, width=1)
            self.dr.text((self.x0 - 10, y), fmt.format(v), font=font(size),
                         fill=GREY, anchor="rm")

    def xlabel(self, text, size=15):
        self.dr.text(((self.x0 + self.x1) / 2, self.y1 + 36), text,
                     font=font(size), fill=GREY, anchor="mt")

    def ylabel(self, text, size=15):
        self.dr.text((self.x0 - 58, (self.y0 + self.y1) / 2), text,
                     font=font(size), fill=GREY, anchor="mm")

    def line(self, xs, ys, color=BLUE, width=3):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            self.dr.line(pts, fill=color, width=width, joint="curve")

    def dot(self, x, y, r=6, fill=BLUE, outline=BG):
        cx, cy = self.px(x), self.py(y)
        self.dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill,
                        outline=outline, width=2)

    def ring(self, x, y, r=9, color=INK, width=2):
        cx, cy = self.px(x), self.py(y)
        self.dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)


def cmap(t):
    """0..1 映射到浅蓝→深蓝。低值浅，高值深。"""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (240, 244, 250)), (0.35, (186, 208, 236)),
             (0.7, (92, 136, 196)), (1.0, (30, 62, 128))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
    return stops[-1][1]


def heat(ax, fn, nx=90, ny=90, vmin=0.0, vmax=1.0):
    """在 ax 的框里逐格填色。fn(x, y) -> 值。"""
    w = (ax.x1 - ax.x0) / nx
    h = (ax.y1 - ax.y0) / ny
    a, b = ax.xlim
    c, d = ax.ylim
    for i in range(nx):
        x = a + (i + 0.5) * (b - a) / nx
        for j in range(ny):
            y = c + (j + 0.5) * (d - c) / ny
            v = (fn(x, y) - vmin) / (vmax - vmin)
            ax.dr.rectangle(
                [ax.x0 + i * w, ax.y0 + j * h, ax.x0 + (i + 1) * w + 1, ax.y0 + (j + 1) * h + 1],
                fill=cmap(v), width=0)


def title(dr, text, sub=None, w=1240):
    dr.text((70, 40), text, font=font(30, True), fill=INK)
    if sub:
        dr.text((70, 84), sub, font=font(15), fill=GREY)
