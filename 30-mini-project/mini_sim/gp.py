"""gp.py —— 手写一个高斯过程回归（只有 numpy）。

用法
----
    python gp.py              # 用 9 个点拟合，出 fig_phase.png，并打印对 25 点的误差
    python gp.py --all        # 用全部 25 点拟合（对照：数据全给时能到多准）

为什么自己写
------------
    本机没有 sklearn、没有 scipy。但高斯过程本身只需要一个 Cholesky 分解，
    60 行就够，而且每一行都能解释——这对"能不能讲清楚"比调包重要。

模型
----
    y(x) = f(x) + ε,   f ~ GP(0, k),   ε ~ N(0, σ_n²)
    k(x, x') = σ_f² exp( -½ Σ_d |x_d - x'_d|² / ℓ_d² )      （各维独立长度尺度）

    后验均值： m(x*) = k*ᵀ (K + σ_n² I)⁻¹ y
    后验方差： s²(x*) = k(x*,x*) - k*ᵀ (K + σ_n² I)⁻¹ k*

    实际实现用 Cholesky：K + σ_n²I = L Lᵀ，然后用两次三角求解代替显式求逆。
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

import viz

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "_scan.json")
FIG = os.path.join(HERE, "fig_phase.png")

NL_MAX = 14.0
EP_MIN, EP_MAX = 12.0, 120.0


# ----------------------------------------------------------------------
# 1. 坐标变换
# ----------------------------------------------------------------------
def to_unit(nlig, eps):
    """把物理参数映射到 [0,1]²。eps 用对数，因为它的作用是对数尺度的。"""
    u = np.asarray(nlig, dtype=float) / NL_MAX
    v = (np.log(np.asarray(eps, dtype=float)) - np.log(EP_MIN)) / (np.log(EP_MAX) - np.log(EP_MIN))
    if u.ndim == 0:
        return np.array([float(u), float(v)])
    return np.stack([u, v], axis=-1)


def from_unit(U):
    U = np.atleast_2d(U)
    nlig = U[:, 0] * NL_MAX
    eps = np.exp(np.log(EP_MIN) + U[:, 1] * (np.log(EP_MAX) - np.log(EP_MIN)))
    return nlig, eps


# ----------------------------------------------------------------------
# 2. 高斯过程
# ----------------------------------------------------------------------
class GPR:
    def __init__(self, lens=(0.45, 0.45), sigma_f=None, sigma_n=0.05):
        self.lens = np.asarray(lens, dtype=float)
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def kernel(self, A, B):
        """RBF 核。A: (n,d), B: (m,d) -> (n,m)"""
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2 / self.lens ** 2).sum(-1)
        return self.sigma_f ** 2 * np.exp(-0.5 * d2)

    def fit(self, X, y):
        self.X = np.atleast_2d(X)
        self.y_mean = float(np.mean(y))
        yc = np.asarray(y, dtype=float) - self.y_mean
        if self.sigma_f is None:
            self.sigma_f = max(float(np.std(y)), 1e-3) * 1.2
        K = self.kernel(self.X, self.X) + self.sigma_n ** 2 * np.eye(len(X))
        # Cholesky 分解；加一点点抖动防止数值上不正定
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
        z = np.linalg.solve(self.L, yc)
        self.alpha = np.linalg.solve(self.L.T, z)
        return self

    def predict(self, Xs, return_var=True):
        Xs = np.atleast_2d(Xs)
        Ks = self.kernel(self.X, Xs)                 # (n, m)
        mu = Ks.T @ self.alpha + self.y_mean
        if not return_var:
            return mu
        v = np.linalg.solve(self.L, Ks)              # (n, m)
        var = self.sigma_f ** 2 - (v ** 2).sum(0)
        return mu, np.maximum(var, 1e-12)


# ----------------------------------------------------------------------
# 3. 读数据
# ----------------------------------------------------------------------
def load_scan(path=SCAN):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = d["rows"]
    nl = np.array([r["nlig"] for r in rows], dtype=float)
    ep = np.array([r["eps"] for r in rows], dtype=float)
    # 吸附深度 = -gap，下限截到 0：脱附的点 gap 会跑到 +7 nm，
    # 直接取 -gap 会得到一个 -7 的悬崖，把整张图拉坏。
    # 物理上"没吸住"就是没有压入，所以记 0。
    y = np.maximum(-np.array([r["gap"] for r in rows], dtype=float), 0.0)
    X = to_unit(nl, ep)
    return d, nl, ep, y, X


def pick_nine(nl, ep):
    """从 25 点里挑 9 个做训练集：三个配体水平 × 三个强度水平，均匀铺开。"""
    idx = []
    for i in (0, 2, 4):
        for j in (0, 2, 4):
            idx.append(i * 5 + j)
    return np.array(idx)


# ----------------------------------------------------------------------
# 4. 画图
# ----------------------------------------------------------------------
def draw(nl, ep, y, X, train_idx, mu_grid, sd_grid, err_txt):
    W, H = 1240, 860
    img = Image.new("RGB", (W, H), viz.BG)
    dr = viz.ImageDraw.Draw(img)
    viz.title(dr, "吸附相图：9 次模拟能复原出什么",
              "颜色 = 预测的吸附深度（nm）；圈出的点是真正跑过的 9 个；小点 = 全部 25 个实测点")

    box = (170, 170, 830, 740)
    L0, L1 = np.log10(12.0), np.log10(120.0)
    ax = viz.Axes(dr, box, (0, 14), (L0, L1))
    mu, sd = mu_grid.reshape(90, 90), sd_grid.reshape(90, 90)
    vmax = max(1.0, float(np.nanmax(mu)))

    # 逐格填色。纵轴用 log(eps)，所以格子是等高的。
    wpx = (box[2] - box[0]) / 90
    hpx = (box[3] - box[1]) / 90
    for i in range(90):
        for j in range(90):
            dr.rectangle([box[0] + i * wpx, box[3] - (j + 1) * hpx,
                          box[0] + (i + 1) * wpx + 1, box[3] - j * hpx + 1],
                         fill=viz.cmap(mu[i, j] / vmax), width=0)
    ax.frame()
    ax.xticks([0, 4, 7, 10, 14])

    # 纵轴刻度：位置在 log(eps)，标签写真实 eps
    for e in (12, 22, 40, 70, 120):
        yy = ax.py(np.log10(e))
        dr.line([(box[0] - 6, yy), (box[0], yy)], fill=viz.LIGHT, width=1)
        dr.text((box[0] - 12, yy), f"{e}", font=viz.font(14), fill=viz.GREY, anchor="rm")

    ax.xlabel("配体条数 nlig")
    dr.text((box[0], box[1] - 34), "吸引强度 eps（纵轴为对数轴）",
            font=viz.font(15), fill=viz.GREY)

    # 25 个实测点
    for k in range(len(nl)):
        ax.dot(nl[k], np.log10(ep[k]), r=4,
               fill=viz.INK if y[k] > 0.05 else viz.RED, outline=viz.BG)
    # 9 个训练点
    for k in train_idx:
        ax.ring(nl[k], np.log10(ep[k]), r=11, color=viz.AMBER, width=3)

    # 色标
    cb_x0, cb_x1 = 856, 878
    cb_y0, cb_y1 = 240, 660
    for k in range(120):
        f = k / 119
        dr.rectangle([cb_x0, cb_y1 - (k + 1) * (cb_y1 - cb_y0) / 120,
                      cb_x1, cb_y1 - k * (cb_y1 - cb_y0) / 120 + 1],
                     fill=viz.cmap(f), width=0)
    dr.rectangle([cb_x0, cb_y0, cb_x1, cb_y1], outline=viz.LIGHT, width=1)
    dr.text((cb_x1 + 10, cb_y1), "0", font=viz.font(14), fill=viz.GREY, anchor="lm")
    dr.text((cb_x1 + 10, cb_y0), f"{vmax:.1f}", font=viz.font(14), fill=viz.GREY, anchor="lm")
    dr.text(((cb_x0 + cb_x1) / 2, cb_y0 - 18), "nm", font=viz.font(13), fill=viz.GREY, anchor="mt")

    # 图例
    lx, ly = 940, 210
    dr.text((lx, ly), "怎么读", font=viz.font(16, True), fill=viz.INK)
    lines = [
        "颜色越深，颗粒陷得越深。",
        "左下：少配体 + 弱吸引，几乎不吸。",
        "右上：多配体 + 强吸引，陷得最深。",
        "中间那片浅色是吸附/脱附的边界。",
        "",
        "橙圈 = 训练用的 9 个点。",
        "黑点 = 吸附住；红点 = 脱附飘走。",
        "",
        err_txt,
    ]
    for i, s in enumerate(lines):
        dr.text((lx, ly + 34 + i * 30), s, font=viz.font(15), fill=viz.GREY)

    img.save(FIG)
    print(f"figure saved: {FIG}")


# ----------------------------------------------------------------------
# 5. 主程序
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="用全部 25 点训练（对照）")
    args = ap.parse_args()

    d, nl, ep, y, X = load_scan()
    idx = np.arange(len(y)) if args.all else pick_nine(nl, ep)

    gp = GPR(sigma_n=0.05).fit(X[idx], y[idx])
    mu, sd = gp.predict(X)
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    mx = float(np.max(np.abs(mu - y)))
    print(f"训练点 {len(idx)} 个，对全部 {len(y)} 点的 RMSE = {rmse:.3f} nm，最大误差 {mx:.3f} nm")

    # 预测网格
    gx = np.linspace(0, 1, 90)
    gy = np.linspace(0, 1, 90)
    U = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    mg, sg = gp.predict(U)

    err_txt = f"9 点拟合：RMSE {rmse:.2f} nm" if not args.all else f"25 点拟合：RMSE {rmse:.2f} nm"
    draw(nl, ep, y, X, idx, mg, sg, err_txt)


if __name__ == "__main__":
    main()
