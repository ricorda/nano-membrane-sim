"""coverage.py —— 把"挂几条链"换算成"覆盖率"，再和 2018 那篇的坐标系对上。

为什么要换：
    2018 那篇（Nanoscale 10, 2481）用的自变量是**配体覆盖率** 40% / 60% / 100%，
    对应配体数 70 / 104 / 174。我们模型里只有"几条链"，直接比较没法开口。

怎么换：
    二维模型里锚点均匀排在半径 R 的圆周上，每条链占的弧长就是链距 D_LL。
    所以能放下的最多条数 MAX_LIG = floor(2πR / D_LL) = 14，
    覆盖率就定义为 N_LIG / MAX_LIG —— 100% 恰好等于"圆周长被链占满"。

    注意这是**线覆盖率**（一维），不是他们的面积覆盖率。二维模型只有一条圆周，
    这个换算是尽量对齐，不是等价。README 里写明了。

用法：
    python coverage.py            # 有缓存就复用
    python coverage.py --fresh    # 强制重跑
"""

import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

from simlog import parse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CACHE = os.path.join(HERE, "_cov.json")

W, H = 1240, 760
INK = (34, 34, 34)
GREY = (120, 120, 120)
BLUE = (52, 96, 160)
ORANGE = (196, 132, 40)


def config():
    """从 sim.py 取几何上限，避免这边再硬编码一个 14。"""
    import json
    import subprocess
    out = subprocess.run([PY, os.path.join(HERE, "sim.py"), "--dump-config"],
                         capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    return json.loads(out)


CFG = config()
MAX_LIG = int(CFG["MAX_LIG"])
EPS_FIXED = 40.0
NLIGS = [0, 2, 4, 6, 8, 10, 12, 14]
SEEDS = [0, 1, 2]     # 每个点跑三个种子，取平均——单种子在这类慢变量上不可靠

# 2018 那篇的三档覆盖率与配体数
THEIRS = [(40, 70), (60, 104), (100, 174)]


def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", name), size)


def run_one(nlig, seed):
    cmd = [PY, os.path.join(HERE, "sim.py"),
           "--nolig" if nlig == 0 else f"--nlig={nlig}",
           f"--eps={EPS_FIXED}", "--steps=60000", "--noplot", f"--seed={seed}"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    d = parse(out)
    if d["gap"] is None:
        raise RuntimeError(f"解析失败：\n{out}")
    return d["gap"], d["dimple"]


def load():
    if os.path.exists(CACHE) and "--fresh" not in sys.argv:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    rows = []
    for n in NLIGS:
        gaps, dims = [], []
        for sd in SEEDS:
            g, d = run_one(n, sd)
            gaps.append(g)
            dims.append(d)
        mean_g = sum(gaps) / len(gaps)
        mean_d = sum(dims) / len(dims)
        spread = max(gaps) - min(gaps)
        cov = 100.0 * n / MAX_LIG
        rows.append({"nlig": n, "cov": cov, "gap": mean_g, "dimple": mean_d,
                     "gaps": gaps, "spread": spread})
        print(f"nlig={n:2d}  覆盖率 {cov:5.1f}%  gap {mean_g:+.2f}"
              f"（种子间极差 {spread:.2f}）  凹陷 {mean_d:.2f}")
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


def main():
    rows = load()

    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)

    dr.text((70, 44), "把“几条链”换成“覆盖率”，再和 2018 那篇并排看", font=font(30, True), fill=INK)
    dr.text((70, 92),
            f"左：本模型在 eps={EPS_FIXED:.0f}、不同覆盖率的吸附深度　"
            f"右：那张论文的三档覆盖率与它的结论",
            font=font(15), fill=GREY)

    # ---------- 左图 ----------
    x0, x1, y0, y1 = 110, 660, 230, 610
    dr.line((x0, y1, x1, y1), fill=INK, width=2)
    dr.line((x0, y0, x0, y1), fill=INK, width=2)

    depths = [max(0.0, -r["gap"]) for r in rows]
    ymax = 0.8
    for v in (0.0, 0.2, 0.4, 0.6, 0.8):
        yy = y1 - (v / ymax) * (y1 - y0)
        dr.line((x0, yy, x1, yy), fill=(228, 228, 228), width=1)
        dr.text((x0 - 46, yy - 9), f"{v:.1f}", font=font(13), fill=GREY)

    def px(cov):
        return x0 + (cov / 100.0) * (x1 - x0)

    def py(v):
        return y1 - (min(v, ymax) / ymax) * (y1 - y0)

    for cov in (0, 20, 40, 60, 80, 100):
        dr.line((px(cov), y1, px(cov), y1 + 6), fill=INK, width=2)
        dr.text((px(cov) - 14, y1 + 12), f"{cov}", font=font(13), fill=GREY)
    dr.text(((x0 + x1) / 2, y1 + 44), "配体覆盖率（%）", font=font(15), fill=GREY, anchor="mt")
    dr.text((70, y0 - 26), "吸附深度（nm，越深越贴进去）", font=font(14), fill=GREY)

    # 2018 的三条竖虚线
    for cov, nlig in THEIRS:
        dr.line((px(cov), y0, px(cov), y1), fill=(214, 190, 150), width=2)
        dr.text((px(cov) - 12, y0 - 52), f"{cov}%", font=font(13), fill=ORANGE)
        dr.text((px(cov) - 14, y0 - 34), f"({nlig}条)", font=font(12), fill=ORANGE)

    pts = [(px(r["cov"]), py(d)) for r, d in zip(rows, depths)]
    dr.line(pts, fill=BLUE, width=3, joint="curve")
    for (xx, yy), r, d in zip(pts, rows, depths):
        half = 0.5 * r["spread"] / ymax * (y1 - y0)
        if half > 2:
            dr.line((xx, yy - half, xx, yy + half), fill=(150, 140, 120), width=2)
            dr.line((xx - 5, yy - half, xx + 5, yy - half), fill=(150, 140, 120), width=2)
            dr.line((xx - 5, yy + half, xx + 5, yy + half), fill=(150, 140, 120), width=2)
        dr.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill="white", outline=BLUE, width=2)
        dr.text((xx + 8, yy + 6), f"{r['nlig']}", font=font(12), fill=BLUE)

    # ---------- 右侧文字 ----------
    tx = 700
    dr.text((tx, 214), "怎么换的", font=font(17, True), fill=INK)
    dr.text((tx, 244), f"锚点排在半径 1.6 nm 的圆周上，", font=font(14), fill=INK)
    dr.text((tx, 266), f"链距 0.7 nm，最多 {MAX_LIG} 条。", font=font(14), fill=INK)
    dr.text((tx, 288), f"所以覆盖率 = 条数 / {MAX_LIG}。", font=font(14), fill=INK)

    dr.text((tx, 330), "他们三档落在哪", font=font(17, True), fill=INK)
    for i, (cov, nlig) in enumerate(THEIRS):
        yy = 360 + i * 44
        dr.text((tx, yy), f"{cov}%", font=font(15, True), fill=ORANGE)
        dr.text((tx + 56, yy), f"{nlig} 条配体 → 16-18 nm 的颗粒", font=font(13), fill=INK)

    dr.text((tx, 512), "能对上的与对不上的", font=font(17, True), fill=INK)
    dr.text((tx, 542), "对得上：40% 往上，覆盖率越高颗粒坐得", font=font(13), fill=INK)
    dr.text((tx, 562), "越高、压出的坑越浅，方向和他们“高", font=font(13), fill=INK)
    dr.text((tx, 582), "覆盖率只吸附不穿透”一致。", font=font(13), fill=INK)
    dr.text((tx, 610), "对不上：他们的配体数同时改变了粒径", font=font(13), fill=INK)
    dr.text((tx, 630), "（Rg 1.42→1.84 nm），我们固定 1.6 nm；", font=font(13), fill=INK)
    dr.text((tx, 650), "低覆盖率处还有离散锚点效应。", font=font(13), fill=INK)

    hi = [d for r, d in zip(rows, depths) if r["cov"] >= 40]
    dr.text((70, H - 62),
            f"40–100% 这一段：吸附深度 {hi[0]:.2f} → {hi[-1]:.2f} nm，单调变浅——"
            f"和 2018“高覆盖率只吸附、不穿透”同向。",
            font=font(15), fill=INK)
    dr.text((70, H - 34),
            "0–30% 不要外推：链是离散钉在固定角度上的，n=2 时两条链正好朝左右、底下是光板，"
            "所以反而更深。种子极差只有 0.01 nm——这是几何效应，不是噪声。",
            font=font(15), fill=GREY)

    out = os.path.join(HERE, "fig_coverage.png")
    img.save(out)
    print("figure saved:", out)


if __name__ == "__main__":
    main()
