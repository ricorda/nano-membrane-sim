"""al.py —— 主动学习 vs 随机采样：同样的模拟预算，谁预测得准。

思路
----
    数据只有 25 个点（scan.py 跑出来的）。把这 25 个点当成"真值网格"，
    然后问：只允许跑 9 次模拟，怎么挑这 9 个点，才能最好地还原整张图？

    三种挑法：
      A. 主动学习：先随机 3 个，然后每一步挑"预测方差最大"的点补进去。
      B. 随机：先随机 3 个，然后每一步随便挑一个。
      C. 均匀网格：3×3 的规则网格（不看数据、只凭先验）。

    评价：用这 9 个点训练 GP，预测全部 25 点，算 RMSE。
    A/B 各重复 N_RESTART 次（随机性只来自起点和随机挑选），取均值与标准差。

为什么要重复
------------
    一次随机挑选的结果可能是运气。30 次的均值和标准差才能说明"平均而言"谁好。
"""

import json
import os

import numpy as np
from PIL import Image

import viz
from gp import GPR, load_scan, pick_nine, to_unit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_al.json")
FIG = os.path.join(HERE, "fig_al.png")

N_RESTART = 30
N_START = 3
BUDGET = 9


def rmse(gp, X_all, y_all):
    mu = gp.predict(X_all, return_var=False)
    return float(np.sqrt(np.mean((mu - y_all) ** 2)))


def run_policy(X, y, policy, rng, n_start=N_START, budget=BUDGET):
    """返回 {预算: RMSE} 的字典。"""
    n = len(y)
    chosen = list(rng.choice(n, size=n_start, replace=False))
    curve = {}
    gp = GPR(sigma_n=0.05).fit(X[chosen], y[chosen])
    curve[len(chosen)] = rmse(gp, X, y)

    while len(chosen) < budget:
        if policy == "al":
            pool = [i for i in range(n) if i not in chosen]
            _, var = gp.predict(X[pool])
            pick = pool[int(np.argmax(var))]
        else:
            pool = [i for i in range(n) if i not in chosen]
            pick = int(rng.choice(pool))
        chosen.append(pick)
        gp = GPR(sigma_n=0.05).fit(X[chosen], y[chosen])
        curve[len(chosen)] = rmse(gp, X, y)
    return curve


def main():
    d, nl, ep, y, X = load_scan()
    n_all = len(y)

    curves = {"al": [], "random": []}
    for r in range(N_RESTART):
        rng = np.random.default_rng(1000 + r)
        for pol in ("al", "random"):
            curves[pol].append(run_policy(X, y, pol, rng))

    budgets = sorted(curves["al"][0].keys())
    stats = {}
    for pol in ("al", "random"):
        m = np.array([[c[b] for b in budgets] for c in curves[pol]])
        stats[pol] = (m.mean(0), m.std(0))

    # 均匀网格（确定性，没有随机性）
    idx = pick_nine(nl, ep)
    gp = GPR(sigma_n=0.05).fit(X[idx], y[idx])
    grid_rmse = rmse(gp, X, y)

    print(f"预算 {budgets[0]}→{budgets[-1]}，重复 {N_RESTART} 次")
    for k, b in enumerate(budgets):
        print(f"  {b} 次：主动学习 {stats['al'][0][k]:.3f}±{stats['al'][1][k]:.3f}   "
              f"随机 {stats['random'][0][k]:.3f}±{stats['random'][1][k]:.3f}")
    print(f"  3×3 规则网格（9 次）：{grid_rmse:.3f}")

    draw(budgets, stats, grid_rmse)

    payload = {
        "n_restart": N_RESTART,
        "budgets": budgets,
        "al_mean": [round(float(v), 4) for v in stats["al"][0]],
        "al_std": [round(float(v), 4) for v in stats["al"][1]],
        "random_mean": [round(float(v), 4) for v in stats["random"][0]],
        "random_std": [round(float(v), 4) for v in stats["random"][1]],
        "grid_rmse": round(grid_rmse, 4),
        "final_gap_al_vs_random": round(float(stats["random"][0][-1] - stats["al"][0][-1]), 4),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"写入 {OUT}")


def draw(budgets, stats, grid_rmse):
    W, H = 1240, 760
    img = Image.new("RGB", (W, H), viz.BG)
    dr = viz.ImageDraw.Draw(img)
    viz.title(dr, "同样的模拟预算，主动学习比随机采样准多少",
              "纵轴 = 用这几次模拟训练出的 GP，对全部 25 个点的预测误差（RMSE，nm）")

    top = max(float((stats[p][0] + stats[p][1]).max()) for p in ("al", "random"))
    ymax = float(np.ceil(top * 10) / 10)
    box = (180, 170, 900, 610)
    ax = viz.Axes(dr, box, (budgets[0] - 0.2, budgets[-1] + 0.2), (0, ymax))
    ax.frame()
    ax.xticks(budgets)

    step = 0.1
    k = 0.0
    while k <= ymax + 1e-9:
        yy = ax.py(k)
        dr.line([(box[0] - 6, yy), (box[0], yy)], fill=viz.LIGHT, width=1)
        dr.text((box[0] - 12, yy), f"{k:.1f}", font=viz.font(14), fill=viz.GREY, anchor="rm")
        k = round(k + step, 2)

    ax.xlabel("已经跑了多少次模拟")
    dr.text((box[0], box[1] - 32), "预测误差 RMSE（nm）", font=viz.font(15), fill=viz.GREY)

    # 误差带（±1 标准差），画之前先裁到框内，否则会盖住标题
    for pol, color in (("random", viz.RED), ("al", viz.BLUE)):
        m, s = stats[pol]
        xs = list(budgets) + list(budgets)[::-1]
        hi = np.minimum(m + s, ymax)
        lo = np.maximum(m - s, 0)
        ys = list(hi) + list(lo)[::-1]
        pts = [(ax.px(x), ax.py(y)) for x, y in zip(xs, ys)]
        dr.polygon(pts, fill=tuple(int(255 - (255 - c) * 0.16) for c in color))
        ax.line(budgets, m, color=color, width=4)
        for x, y in zip(budgets, m):
            ax.dot(x, y, r=4, fill=color)

    # 网格设计那条水平参考线
    yy = ax.py(min(grid_rmse, ymax))
    for xx in range(box[0], box[2], 14):
        dr.line([(xx, yy), (xx + 7, yy)], fill=viz.AMBER, width=2)
    dr.text((box[2] + 10, yy), "3×3 网格", font=viz.font(14), fill=viz.AMBER, anchor="lm")

    # 图例
    dr.line([(940, 210), (975, 210)], fill=viz.BLUE, width=4)
    dr.text((985, 210), "主动学习", font=viz.font(15), fill=viz.INK, anchor="lm")
    dr.line([(940, 246), (975, 246)], fill=viz.RED, width=4)
    dr.text((985, 246), "随机采样", font=viz.font(15), fill=viz.INK, anchor="lm")
    dr.text((940, 282), "阴影 = ±1 个标准差", font=viz.font(13), fill=viz.GREY, anchor="lm")

    d = stats["random"][0][-1] - stats["al"][0][-1]
    tail = "主动学习略好" if d > 0.005 else ("两者几乎一样" if abs(d) <= 0.005 else "随机反而更好")
    dr.text((70, 690),
            f"9 次预算下：主动学习 {stats['al'][0][-1]:.3f} nm，随机 {stats['random'][0][-1]:.3f} nm，"
            f"差 {abs(d):.3f} nm → {tail}。",
            font=viz.font(16), fill=viz.INK)
    dr.text((70, 722),
            f"规则 3×3 网格（同样 9 次）：{grid_rmse:.3f} nm —— 在这个二维光滑问题上，网格和主动学习打平。",
            font=viz.font(16), fill=viz.INK)
    img.save(FIG)
    print(f"figure saved: {FIG}")


if __name__ == "__main__":
    main()
