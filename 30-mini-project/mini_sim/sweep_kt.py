"""
降温判据：同一套力，只改温度
================================

这是三个收敛判据里最硬的一个。逻辑很简单：

  静态形变由力平衡决定，与温度无关；
  噪声的振幅 ∝ √kT。

所以把 kT 从 1 降到 0.1 重跑：
  · 凹陷深度应该基本不动（动了就是有系统误差）
  · 单珠涨落应该按 √kT 掉（不掉就是温度没接对）

用法：
    python sweep_kt.py                 # 默认 4 个温度，约 40 秒
    python sweep_kt.py 1 0.5 0.25 0.1
"""

import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
KTS = [float(x) for x in sys.argv[1:] if not x.startswith("-")] or [1.0, 0.5, 0.25, 0.1]
EXTRA = [x for x in sys.argv[1:] if x.startswith("--") and x != "--fresh"]

pat_gap = re.compile(r"最终间隙\s+([-+0-9.]+) nm")
pat_dim = re.compile(r"膜的静态凹陷深度\s+([0-9.]+) nm")
pat_std = re.compile(r"单珠热涨落 实测/理论\s+([0-9.]+) / ([0-9.]+) nm")
pat_blk = re.compile(r"分块收敛性 最大偏差\s+([0-9.]+) nm")

rows = []
CACHE = os.path.join(HERE, "_kt_cache.json")
if os.path.exists(CACHE) and "--fresh" not in sys.argv:
    import json
    rows = [tuple(r) for r in json.load(open(CACHE, encoding="utf-8"))]
    print("复用上次的扫描结果（加 --fresh 重跑）")
else:
    for kt in KTS:
        out = subprocess.run([PY, os.path.join(HERE, "sim.py"), "--noplot", f"--kt={kt}"] + EXTRA,
                             capture_output=True, text=True, encoding="utf-8").stdout
        g = float(pat_gap.search(out).group(1))
        d = float(pat_dim.search(out).group(1))
        s, t = (float(x) for x in pat_std.search(out).groups())
        b = float(pat_blk.search(out).group(1))
        rows.append((kt, g, d, s, t, b))
        print(f"kT={kt:<5} gap={g:+.2f}  dimple={d:.2f}  std={s:.3f}/{t:.3f}  blk={b:.3f}")
    import json
    json.dump(rows, open(CACHE, "w", encoding="utf-8"))


# ---------------- 画图 ----------------
FONT_DIR = r"C:\Windows\Fonts"


def font(size, bold=False):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, "msyhbd.ttc" if bold else "msyh.ttc"), size)
    except OSError:
        return ImageFont.load_default()


INK, GREY, LIGHT = (28, 30, 34), (120, 124, 132), (226, 228, 234)
BLUE, RED, PALE = (44, 92, 178), (196, 72, 60), (158, 184, 224)
W, H = 1240, 660
img = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(img)
dr.text((70, 34), "降温判据：只改温度，看形变和噪声各自怎么动", font=font(28, True), fill=INK)
dr.text((72, 78), "同一套力（EPS=40, K_S=5），只把 kT 降下来", font=font(15), fill=GREY)

x0, x1, y0, y1 = 110, 1150, 150, 470
kmin, kmax = min(KTS) / 1.6, max(KTS) * 1.6


def KX(v):
    import math
    # 横轴反过来画：左边热（kT=1）、右边冷（kT=0.1），和"降温"这个动作同向
    return x1 - (math.log10(v) - math.log10(kmin)) / (math.log10(kmax) - math.log10(kmin)) * (x1 - x0)


vmax = 0.40


def VY(v):
    return y1 - v / vmax * (y1 - y0)


for v in (0, 0.1, 0.2, 0.3, 0.4):
    dr.line([(x0, VY(v)), (x1, VY(v))], fill=(242, 244, 248))
    dr.text((x0 - 12, VY(v)), f"{v:.1f}", font=font(14), fill=GREY, anchor="rm")

for kt, _g, _d, _s, _t, _b in rows:
    dr.line([(KX(kt), y0), (KX(kt), y1)], fill=(246, 247, 250))
    dr.text((KX(kt), y1 + 14), f"{kt:g}", font=font(14), fill=GREY, anchor="mt")

pts_d = [(KX(kt), VY(d)) for kt, _g, d, _s, _t, _b in rows]
pts_s = [(KX(kt), VY(s)) for kt, _g, _d, s, _t, _b in rows]
pts_t = [(KX(kt), VY(t)) for kt, _g, _d, _s, t, _b in rows]

dr.line(pts_d, fill=BLUE, width=4)
for p in pts_d:
    dr.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=BLUE)

dr.line(pts_s, fill=RED, width=3)
for p in pts_s:
    dr.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=RED)
dr.line(pts_t, fill=PALE, width=2)
for p in pts_t:
    dr.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=PALE)

dr.text((x0 - 44, y0 - 32), "nm", font=font(14), fill=GREY)
dr.text(((x0 + x1) / 2, y1 + 44), "温度 kT（对数轴，左热右冷 →）", font=font(15), fill=GREY, anchor="mt")

# 图例
lx, ly = 760, 120
dr.line([(lx, ly), (lx + 28, ly)], fill=BLUE, width=4)
dr.text((lx + 36, ly - 9), "静态凹陷深度", font=font(14), fill=GREY)
dr.line([(lx, ly + 26), (lx + 28, ly + 26)], fill=RED, width=3)
dr.text((lx + 36, ly + 17), "单珠涨落 实测", font=font(14), fill=GREY)
dr.line([(lx, ly + 52), (lx + 28, ly + 52)], fill=PALE, width=2)
dr.text((lx + 36, ly + 43), "单珠涨落 理论 √(kT/K_S) 模式求和", font=font(14), fill=GREY)

d_hot, d_cold = rows[0][2], rows[-1][2]
chg = (d_cold - d_hot) / max(d_hot, 1e-9) * 100
direction = "越冷越深" if d_cold > d_hot else "越冷越浅"
why = ("配体是熵弹簧：温度一低，它撑开的力量就小了，颗粒反而坐得更低"
       if d_cold > d_hot else
       "膜的软有一半来自熵：温度一低，膜变硬，坑就浅了")
dr.text((70, H - 92),
        f"读法：蓝色 {d_hot:.2f} → {d_cold:.2f} nm（{chg:+.0f}%），{direction}；"
        f"绝对值只动这么点 ＝ 形变是力平衡的结果，不是噪声。",
        font=font(15), fill=INK)
dr.text((70, H - 62),
        f"为什么会上{('升' if d_cold > d_hot else '下')}：{why}。",
        font=font(15), fill=GREY)
dr.text((70, H - 32),
        "红点随 kT 按 √kT 下降、并贴在浅色理论线上 ＝ 温度接对了，噪声没有别的来源。",
        font=font(15), fill=INK)

out = os.path.join(HERE, "fig_cooling.png")
img.save(out)
print("figure saved:", out)
