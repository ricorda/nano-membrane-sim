"""
二维布朗动力学：纳米颗粒在脂质膜上的吸附
================================================

这个脚本不依赖任何分子模拟软件，只用 numpy 和 PIL。
它把连续的真实体系砍成最简单的两维、几百个自由度，
目的不是"预测真实材料"，而是让你看清吸附这件事是怎么由力和涨落决定的。

几何约定
--------
  x  : 沿膜面方向，单位 nm。珠子固定在晶格点上，只能在 z 方向动。
  z  : 垂直于膜面，向上为正。

膜    = 上下两条珠子链（上叶、下叶），各 64 个珠子，间距 1.0 nm，
        平衡厚度 2.4 nm。珠子被三组弹簧拉住：
          ① 拴在晶格点上的弹簧 K_S   —— 决定膜有多"软"
          ② 相邻珠子之间的弹簧 K_B   —— 决定弯曲的代价
          ③ 上下两叶之间的排斥 K_T   —— 只在下叶被压近时才起作用

颗粒  = 一个硬圆盘，半径 R = 1.6 nm，只在 z 方向动（x 固定在膜中央）。
        它与珠子之间有两部分作用：
          ① 硬核排斥 K_CORE —— 珠子不能钻进颗粒里
          ② 指数吸引壳 EPS  —— 距离超过 R 之后按 exp(-(d-R)/W) 衰减，
             在 R+CUT 之前用光滑开关降到零。这就是"黏性"的来源。
             注意：是"指数"不是"高斯"——两者在 d=R 处的行为正好相反，
             指数在接触点力最大，高斯在接触点力为零。

配体  = 颗粒表面挂的短链，默认 10 条、每条 4 颗珠子。
        它们只做几何：链上的键 + 珠子之间的排除体积 + 和膜的软墙排斥。
        没有额外的吸引——因为配体是亲水的柔性链，趴在膜上要损失构象熵，
        净贡献是"占地方"，不是"黏住"。真实的黏附来自金核和膜的范德华吸引。
        锚点固定（颗粒不转），配体珠子在二维平面里自由移动。

运动  = 过阻尼朗之万：每一步  x += F*dt + sqrt(2*dt*kT)*xi
        没有质量、没有惯性，力大就流得快。xi 是标准差为 1 的高斯白噪声。
        温度只从这个噪声项进入，力的公式里不出现 kT。

运行
----
    python sim.py                    # 默认参数，出图
    python sim.py --eps=5            # 把吸引强度调大
    python sim.py --nolig            # 光板球，看没有配体时吸附多深
    python sim.py --nlig=14          # 配体最密（二维圆盘的几何上限）
    python sim.py --steps=200000     # 跑更久
    python sweep_kt.py               # 降温判据那张图
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------
# 0. 命令行参数
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="二维布朗动力学：颗粒在膜上的吸附")
    p.add_argument("--eps", type=float, default=40.0, help="吸引强度（每个珠子），默认 40")
    p.add_argument("--steps", type=int, default=120_000, help="总步数，默认 120000")
    p.add_argument("--gap", type=float, default=2.0,
                   help="颗粒初始间隙 nm，默认 2.0（要小于 吸引截断-半径，否则起步时还没有相互作用）")
    p.add_argument("--ks", type=float, default=5.0, help="膜的软硬（拴在晶格点的弹簧刚度），默认 5")
    p.add_argument("--seed", type=int, default=1, help="随机种子")
    p.add_argument("--twoleaf", action="store_true",
                   help="让下叶也被同一个吸引壳拉住（旧做法，默认不用）。"
                        "默认只让近侧的上叶受吸引：颗粒从上方靠近，隔着 2.4 nm 的疏水核心，"
                        "远端那一叶不该吃到同一个范德华吸引")
    p.add_argument("--oneleaf", action="store_true",
                   help="（兼容旧写法）默认已经是单叶，这个开关留着不报错")
    p.add_argument("--blocks", type=int, default=4, help="把后半段切成几块做收敛性检查")
    p.add_argument("--nlig", type=int, default=10,
                   help="颗粒表面挂几条配体链，默认 10；给 0 就是光板球")
    p.add_argument("--nolig", action="store_true", help="完全关掉配体（等价于 --nlig=0）")
    p.add_argument("--kt", type=float, default=1.0,
                   help="温度（约化单位，默认 1）。调小它：静态形变应该不变，噪声按 √kT 变小")
    p.add_argument("--noplot", action="store_true", help="只算不画图")
    p.add_argument("--dump-config", dest="dump_config", action="store_true",
                   help="把几何与力学的关键常数打印成 JSON 后退出（给别的脚本读数用）")
    return p.parse_args()


ARGS = get_args()

# ----------------------------------------------------------------------
# 1. 模型参数（单位：nm，能量以 kT 为单位）
# ----------------------------------------------------------------------
N_COL = 64          # 每条链的珠子数
A = 1.0             # 珠子间距 nm
D0 = 2.4            # 膜的平衡厚度 nm

K_S = ARGS.ks        # 珠子和晶格点之间的弹簧刚度（越小越软）
K_B = 100.0         # 相邻珠子之间的弹簧刚度（抵抗弯曲）
K_T = 30.0          # 上下两叶之间的排斥刚度（抵抗变薄）

R = 1.6             # 颗粒半径 nm
K_CORE = 120.0      # 颗粒硬核排斥刚度（越大越硬）
EPS = ARGS.eps      # 吸引强度
W = 0.9             # 吸引衰减长度 nm
CUT = 3 * W         # 吸引截断距离（相对颗粒表面）

DT = 0.002          # 时间步长
KT = ARGS.kt        # 温度（约化单位）。力的公式里不出现它，只有涨落项用它
STEP_MAX = 0.20     # 单步最大位移，防止爆炸（要比 σ=√(2·dt·kT) 大 3 倍以上，否则会截断噪声）
P_MOB = 0.6         # 颗粒的迁移率因子（颗粒比珠子大，动得慢）
REC = 20            # 每隔多少步记录一次
FAR_NM = 5.0        # 离颗粒多远算"远处膜面"（吸引截断 R+CUT ≈ 4.3 nm，留余量）
Z_CEIL = D0 / 2 + R + 8.0   # 颗粒中心的上限，对应 gap = +8 nm（撞到它就是脱附）

# ---- 配体链（只做几何：占地方，不和膜发生吸引）----
L_LIG = 4           # 每条链几颗珠子
L_BOND = 0.7        # 链上相邻珠子的平衡距离 nm
R_LIG = 0.35        # 配体珠子半径 nm
D_LL = 2 * R_LIG    # 配体珠子之间的接触距离
D_LM = 0.9          # 配体珠子和膜珠子的接触距离（0.35 + 0.55）
K_LIGB = 50.0       # 链上键的刚度（配体是熵弹簧，比共价键软得多）
K_LIGX = 30.0       # 排除体积的刚度

SWITCH = 0.6        # 吸引势在 [R+CUT-0.6, R+CUT] 区间内平滑关掉的宽度

N_LIG = 0 if ARGS.nolig else ARGS.nlig   # 挂几条链
# 几何上限：二维里锚点排在半径 R 的圆周上，两根链的锚点不能比珠子直径更近。
MAX_LIG = max(1, int(np.floor(2 * np.pi * R / D_LL)))
if N_LIG > MAX_LIG:
    print(f"注意：二维圆盘上最多放得下 {MAX_LIG} 条链"
          f"（周长 {2*np.pi*R:.1f} nm ÷ 链距 {D_LL:.1f} nm），已按 {MAX_LIG} 处理。"
          f"真实三维里同一面积能放更多，这是二维模型的局限。")
    N_LIG = MAX_LIG

if ARGS.dump_config:
    # 别的脚本要跟这里保持同一个"事实来源"，别各自硬编码一份
    print(json.dumps({
        "R": R, "D_LL": D_LL, "D_LM": D_LM, "D0": D0, "A": A, "N_COL": N_COL,
        "K_S": K_S, "K_B": K_B, "K_T": K_T, "KT": KT, "STEP_MAX": STEP_MAX,
        "W": W, "CUT": CUT, "SWITCH": SWITCH, "L_LIG": L_LIG, "L_BOND": L_BOND,
        "MAX_LIG": MAX_LIG, "FAR_NM": FAR_NM, "Z_CEIL": Z_CEIL,
        "twoleaf": bool(ARGS.twoleaf),
    }, ensure_ascii=False))
    raise SystemExit(0)

L = N_COL * A       # 膜的总长度 nm
# 颗粒的 x 位置：放在晶格的正中，即两根珠子之间。
# 若放在某根珠子正上方（如 32.0），左右两侧的珠子数会差一个，形变就会左右不对称。
PX = (N_COL - 1) * A / 2

SEED = ARGS.seed
STEPS = ARGS.steps
GAP0 = ARGS.gap


# ----------------------------------------------------------------------
# 2. 初始化：一张平膜，一个悬在上方的颗粒
# ----------------------------------------------------------------------
rng = np.random.default_rng(SEED)

x_grid = np.arange(N_COL) * A                        # 珠子的 x 坐标（固定）
z_u0 = np.full(N_COL, D0 / 2)                        # 上叶平衡高度
z_d0 = np.full(N_COL, -D0 / 2)                       # 下叶平衡高度

z_u = z_u0 + 0.02 * rng.standard_normal(N_COL)       # 上叶实际高度
z_d = z_d0 + 0.02 * rng.standard_normal(N_COL)       # 下叶实际高度

pz = D0 / 2 + R + GAP0                               # 颗粒中心高度
px = PX


# ----------------------------------------------------------------------
# 3. 力：膜自己 + 颗粒与膜之间
# ----------------------------------------------------------------------
def membrane_forces(z_u, z_d):
    """返回两条链各自受到的力（只算膜自身的弹簧，不含颗粒）。

    ① 拴在晶格点上：F = -K_S * (z - z0)
    ② 相邻珠子：弯曲会让拉伸弹簧变形，F_i += -K_B*(2z_i - z_{i-1} - z_{i+1})
    ③ 上下两叶：只有被压到比 D0 更薄时才排斥，推力把两叶推开
    """
    fu = -K_S * (z_u - z_u0)
    fd = -K_S * (z_d - z_d0)

    # 链上的弯曲（二阶差分）。首尾各少一个邻居。
    fu[1:-1] += -K_B * (2 * z_u[1:-1] - z_u[:-2] - z_u[2:])
    fd[1:-1] += -K_B * (2 * z_d[1:-1] - z_d[:-2] - z_d[2:])
    fu[0] += -K_B * (z_u[0] - z_u[1])
    fu[-1] += -K_B * (z_u[-1] - z_u[-2])
    fd[0] += -K_B * (z_d[0] - z_d[1])
    fd[-1] += -K_B * (z_d[-1] - z_d[-2])

    # 上下两叶的单边排斥：厚度 Δ 小于 D0 时才用力
    thick = z_u - z_d
    push = np.where(thick < D0, K_T * (D0 - thick), 0.0)
    fu += push
    fd -= push

    return fu, fd


def particle_forces(z_u, z_d, pz):
    """返回 (上叶受力, 下叶受力, 颗粒受力, 接触珠子数)。

    珠子相对颗粒的矢量 d = r_bead - r_p，距离 s = |d|。
      s < R          : 硬核排斥，把珠子往外推，颗粒受到相反方向的力
      R <= s < R+CUT : 指数吸引，把珠子往颗粒拉，颗粒被往膜拉
    每个珠子都同时给膜和颗粒加力，保证作用力与反作用力成对出现。
    """
    dz_u = z_u - pz
    dz_d = z_d - pz
    dx = x_grid - px
    dist_u = np.hypot(dx, dz_u)
    dist_d = np.hypot(dx, dz_d)

    f_u = np.zeros(N_COL)
    f_d = np.zeros(N_COL)
    fp_z = 0.0
    n_contact = 0

    for leaf, (dist, dz, f_out) in enumerate(
            ((dist_u, dz_u, f_u), (dist_d, dz_d, f_d))):
        safe = np.maximum(dist, 1e-9)
        ux = dx / safe                 # 从颗粒指向珠子的单位矢量
        uz = dz / safe

        # ① 硬核排斥（两叶都算：任何珠子都不能钻进颗粒里）
        inside = dist < R
        if np.any(inside):
            mag = K_CORE * (R - dist[inside])
            # 只取 z 分量：珠子只能在 z 方向动，x 分量没有对应的自由度。
            # （早先这里多写了一行 mag*ux，把 x 分量叠进了 z 力，导致左右不对称。）
            f_out[inside] += mag * uz[inside]
            fp_z += -np.sum(mag * uz[inside])

        # ② 指数吸引（默认只作用在上叶 0；--twoleaf 时下叶也吃同一份吸引）
        if not ARGS.twoleaf and leaf == 1:
            continue
        shell = (dist >= R) & (dist < R + CUT)
        if np.any(shell):
            d = dist[shell]
            e = np.exp(-(d - R) / W)
            # 平滑截断：在 [d1, d2] 之间把势乘一个光滑开关 s(d)，
            # s 从 1 降到 0，且导数在两端都为零 → 力和能量在截断处都连续。
            # 原来的写法在 d=R+CUT 处力突然跳到零，会让颗粒"踩空"。
            d1, d2 = R + CUT - SWITCH, R + CUT
            t = np.clip((d2 - d) / (d2 - d1), 0.0, 1.0)
            s = t * t * (3 - 2 * t)                # 1 → 0，两端导数都为 0
            # 只把"力的大小"乘上 s，不额外加 s 的导数项：
            # 这个吸引壳本来就是"力随距离指数衰减"的经验形式，不是某个势的梯度，
            # 所以能保证的是力连续（颗粒不会在截断处被突然放开），而不是能量连续。
            mag = EPS * e * s
            f_out[shell] += -mag * uz[shell]
            fp_z += np.sum(mag * uz[shell])
            n_contact += int(np.sum(dist[shell] < R + 0.35))

    return f_u, f_d, fp_z, n_contact


# ----------------------------------------------------------------------
# 3.5 配体链：只做几何
# ----------------------------------------------------------------------
# 为什么"只占地方、不吸引"是有道理的：
#   配体接在金核上，真正把颗粒拽向膜的是金核和膜的范德华吸引；
#   而配体是亲水的柔性链，它趴在膜上时损失构象熵，
#   所以它的净贡献是"排斥"（被压扁要付熵的代价），不是吸引。
#   2013 那篇 SCFT 算的就是这件事：链越长、越密，颗粒越难穿透。

ROOT = np.stack([np.cos(2 * np.pi * (np.arange(N_LIG) + 0.5) / max(N_LIG, 1)),
                 np.sin(2 * np.pi * (np.arange(N_LIG) + 0.5) / max(N_LIG, 1))],
                axis=1) * R if N_LIG else np.zeros((0, 2))


def init_ligands():
    """配体珠相对颗粒中心的位置，初始时沿径向伸直。"""
    lig = np.zeros((N_LIG * L_LIG, 2))
    for k in range(N_LIG):
        u = ROOT[k] / R
        for i in range(L_LIG):
            lig[k * L_LIG + i] = ROOT[k] + u * L_BOND * (i + 1)
    return lig


def ligand_forces(lig, z_u, z_d, pz):
    """返回 (配体受力, 上叶受力, 下叶受力, 颗粒受到的反作用, 接触统计)。

    lig 存的是相对颗粒中心的坐标；膜珠子在实验室坐标里，
    所以算配体-膜相互作用时要先加上 (px, pz)。
    """
    if N_LIG == 0:
        return (np.zeros((0, 2)), np.zeros(N_COL), np.zeros(N_COL), 0.0, 0)

    F = np.zeros_like(lig)

    # ① 链上的键 + 锚点
    fp_z = 0.0
    for k in range(N_LIG):
        idx = np.arange(k * L_LIG, (k + 1) * L_LIG)
        a, b = lig[idx[:-1]], lig[idx[1:]]
        d = b - a
        L = np.linalg.norm(d, axis=1)
        u = d / np.maximum(L, 1e-9)[:, None]
        mag = K_LIGB * (L - L_BOND)
        F[idx[:-1]] += mag[:, None] * u
        F[idx[1:]] -= mag[:, None] * u

        # 锚点在颗粒表面，固定不动（颗粒不转）。它对 bead0 的力：
        d0 = lig[idx[0]] - ROOT[k]
        L0 = float(np.linalg.norm(d0))
        u0 = d0 / max(L0, 1e-9)
        mag0 = K_LIGB * (L0 - L_BOND)
        F[idx[0]] -= mag0 * u0
        # 链对颗粒的反作用（锚点受到的力取反）
        fp_z += mag0 * u0[1]

    # ② 配体珠子之间的排除体积（同一根链上相邻的两颗跳过，它们由键管）
    if len(lig) > 1:
        dd = lig[:, None, :] - lig[None, :, :]
        rr = np.sqrt(np.sum(dd ** 2, axis=2))
        np.fill_diagonal(rr, 1e9)
        for k in range(N_LIG):          # 跳过同链相邻
            for i in range(L_LIG - 1):
                p, q = k * L_LIG + i, k * L_LIG + i + 1
                rr[p, q] = rr[q, p] = 1e9
        hit = rr < D_LL
        if np.any(hit):
            ui, uj = np.where(hit)
            mag = K_LIGX * (D_LL - rr[ui, uj])
            uu = dd[ui, uj] / np.maximum(rr[ui, uj], 1e-9)[:, None]
            np.add.at(F, ui, mag[:, None] * uu)
            np.add.at(F, uj, -mag[:, None] * uu)

    # ③ 配体珠子 vs 膜：把膜当成一张"有厚度的软墙"，而不是一排离散的点。
    #    原因：珠子间距 1.0 nm > 配体直径 0.7 nm，配体会从两个膜珠子中间漏过去。
    #    这里用线性插值求出配体脚下膜的高度，再按垂直距离排斥，
    #    反作用按 (1-w, w) 摊到相邻两颗膜珠子上，力仍然成对。
    ax = px + lig[:, 0]          # 配体珠子的实验室坐标
    az = pz + lig[:, 1]
    f_u = np.zeros(N_COL)
    f_d = np.zeros(N_COL)
    n_contact = 0

    # 线性插值：配体脚下两条叶各自的高度
    xc = np.clip(ax / A, 0, N_COL - 1 - 1e-9)
    i0 = np.floor(xc).astype(int)
    w = xc - i0
    zu_m = (1 - w) * z_u[i0] + w * z_u[i0 + 1]
    zd_m = (1 - w) * z_d[i0] + w * z_d[i0 + 1]

    def _book(rows, mags, leaf):
        """把配体受到的反作用摊回相邻两颗膜珠子（mag 已带符号）。"""
        f_leaf = f_u if leaf == 0 else f_d
        np.add.at(f_leaf, i0[rows], -(1 - w[rows]) * mags)
        np.add.at(f_leaf, i0[rows] + 1, -w[rows] * mags)

    # ③a 在膜的外侧：软壳排斥（两侧对称）
    for leaf, (z_m, sgn_allowed) in enumerate(((zu_m, +1), (zd_m, -1))):
        dzm = az - z_m
        over = D_LM - np.abs(dzm)
        hit = (over > 0) & (np.sign(dzm) == sgn_allowed)
        if not np.any(hit):
            continue
        mag = K_LIGX * over[hit] * sgn_allowed
        F[hit, 1] += mag
        _book(np.where(hit)[0], mag, leaf)
        if leaf == 0:
            # 注意：这里要数"几颗珠子挨着膜"，不是把压入深度加起来。
            n_contact = int(np.count_nonzero(hit))

    # ③b 已经陷进两层膜之间：往更近的那一侧推出去。
    #     颗粒吸附时下半球的锚点本来就在膜里，这条必须处理，否则配体会"穿过"膜。
    #     这里只给配体加力、不给膜反作用：膜珠子被导轨拴住，
    #     而"把一颗珠子从膜内部挤出来"是我们对模型缺陷打的补丁，不是真实的物理力。
    inside = (az > zd_m) & (az < zu_m)
    if np.any(inside):
        rows = np.where(inside)[0]
        d_up = zu_m[rows] - az[rows]
        d_dn = az[rows] - zd_m[rows]
        go_up = d_up <= d_dn
        mags = K_LIGX * (D_LM + np.where(go_up, d_up, d_dn))
        F[rows, 1] += np.where(go_up, mags, -mags)
        # 反作用必须给膜，否则配体子系被"凭空喂力"，
        # 会通过锚点把颗粒持续往上顶（20 条链时颗粒会以 +160 nm 飞走）。
        up_rows, dn_rows = rows[go_up], rows[~go_up]
        if len(up_rows):
            _book(up_rows, mags[go_up], 0)      # 向上挤 → 上叶被压下去
        if len(dn_rows):
            _book(dn_rows, -mags[~go_up], 1)    # 向下挤 → 下叶被顶起来

    # ④ 配体珠子不能钻进颗粒的硬核
    rl = np.linalg.norm(lig, axis=1)
    inside = rl < R + R_LIG
    if np.any(inside):
        uu = lig[inside] / np.maximum(rl[inside], 1e-9)[:, None]
        mag = K_LIGX * (R + R_LIG - rl[inside])
        F[inside] += mag[:, None] * uu

    return F, f_u, f_d, fp_z, n_contact


# ----------------------------------------------------------------------
# 4. 主循环
# ----------------------------------------------------------------------
def run():
    global z_u, z_d, pz

    hist_t, hist_gap, hist_nc = [], [], []
    n_clip = 0                     # 颗粒单步位移被截断的次数（截断会压低有效温度）
    n_ceil = 0                     # 颗粒撞到顶盖的次数（>0 说明脱附了）
    prof_u = np.zeros(N_COL)       # 后半段的时间平均形状，压掉热噪声
    prof_n = 0
    # 收敛性检查：把后半段切成 NBLK 块，各自平均，看它们是否重合
    NBLK = max(1, ARGS.blocks)
    blk_sum = np.zeros((NBLK, N_COL))
    blk_n = np.zeros(NBLK, dtype=int)
    far_samp = []
    T_TOTAL = STEPS * DT
    lig = init_ligands()           # 配体珠子的相对坐标（颗粒不转，锚点固定）
    lig_contact_acc = 0.0          # 后半段"配体压到膜上"的珠子数，时间平均
    lig_contact_n = 0

    for step in range(STEPS + 1):
        # ---- 膜自身的力 ----
        fu, fd = membrane_forces(z_u, z_d)

        # ---- 颗粒与膜之间的力 ----
        dfu, dfd, fp_z, n_contact = particle_forces(z_u, z_d, pz)
        fu += dfu
        fd += dfd

        # ---- 配体链的力（只做几何：占地方、不和膜吸引）----
        flig, lfu, lfd, lfp, nc_lig = ligand_forces(lig, z_u, z_d, pz)
        fu += lfu
        fd += lfd
        fp_z += lfp

        # ---- 积分（过阻尼朗之万）----
        # 涨落振幅 √(2·dt·kT)：力的公式里没有温度，温度只从这里进来。
        noise = np.sqrt(2 * DT * KT)
        z_u += DT * fu + noise * rng.standard_normal(N_COL)
        z_d += DT * fd + noise * rng.standard_normal(N_COL)

        if N_LIG:
            lig += DT * flig + noise * rng.standard_normal(lig.shape)

        dp = P_MOB * (DT * fp_z + noise * rng.standard_normal())
        if abs(dp) > STEP_MAX:
            n_clip += 1
        dp = np.clip(dp, -STEP_MAX, STEP_MAX)
        pz += dp

        # 顶盖：颗粒不能飘出盒子。没有它，弱吸引时颗粒会一路扩散到 +200 nm，
        # 扫参时那些点就变成没有意义的大数。撞到顶盖 = 脱附，记录为 +Z_CEIL。
        if pz > Z_CEIL:
            pz = Z_CEIL
            n_ceil += 1

        # ---- 记录 ----
        if step % REC == 0:
            t = step * DT
            far = np.abs(x_grid - px) > FAR_NM
            base = float(np.median(z_u[far])) if far.sum() > 5 else float(np.median(z_u))
            gap = (pz - R) - base          # 颗粒底面到远处膜面的距离
            hist_t.append(t)
            hist_gap.append(gap)
            hist_nc.append(n_contact)

            # 只对后半段（已经吸附住之后）做时间平均
            if t > 0.6 * T_TOTAL:
                prof_u += z_u
                prof_n += 1
                frac = (t - 0.6 * T_TOTAL) / (0.4 * T_TOTAL)
                k = min(int(frac * NBLK), NBLK - 1)
                blk_sum[k] += z_u
                blk_n[k] += 1
                far_samp.append(z_u[far].copy())
                lig_contact_acc += nc_lig
                lig_contact_n += 1

    prof = prof_u / max(prof_n, 1)
    blocks = np.array([blk_sum[k] / max(blk_n[k], 1) for k in range(NBLK)])
    return (np.array(hist_t), np.array(hist_gap),
            np.array(hist_nc),
            prof, z_u.copy(), pz, blocks, np.array(far_samp),
            n_clip / max(STEPS, 1),
            n_ceil / max(STEPS, 1),
            lig.copy(), lig_contact_acc / max(lig_contact_n, 1))


t_arr, gap_arr, nc_arr, z_u_mean, z_u_fin, pz_fin, blk_mean, far_samp, clip_frac, \
    ceil_frac, lig_fin, lig_contact_mean = run()

# 时间平均形状：先扣掉远处膜面的整体倾斜，再相对远处高度归零
far_mask = np.abs(x_grid - px) > FAR_NM
if far_mask.sum() > 5:
    coef = np.polyfit(x_grid[far_mask], z_u_mean[far_mask], 1)
    plane = np.polyval(coef, x_grid)
else:
    plane = np.full(N_COL, float(np.median(z_u_mean)))
shape = z_u_mean - plane

if os.environ.get("SIM_DEBUG"):
    print("--- 时间平均形状（相对自由膜面）---")
    for i in range(N_COL):
        print(f"x={x_grid[i]:5.1f}  shape={shape[i]:+.3f}")

# ----------------------------------------------------------------------
# 5. 判读结果
# ----------------------------------------------------------------------
final_gap = float(np.mean(gap_arr[-max(1, len(gap_arr) // 4):]))   # 末段 1/4 平均
max_nc = int(np.max(nc_arr))
contact_idx = np.where(gap_arr < 0.3)[0]
contact_time = float(t_arr[contact_idx[0]]) if len(contact_idx) else None

# 静态形变：看时间平均后的膜形状，而不是某一帧的瞬时值
near_mask = np.abs(x_grid - px) <= 2.5
dimple = float(-shape[near_mask].min())            # 颗粒正下方被压下去的深度
ridge = float(shape[near_mask].max())              # 周围被拉起来的高度

# 诊断一：单珠热涨落的实测值 vs 理论值（正常模式均分求和）。
def bead_std_theory(n=N_COL, ks=K_S, kb=K_B, kt=KT):
    """单个膜珠 z 方向涨落的理论值（正常模式的均分求和）。

    能量的傅里叶形式：E = ½ Σ_q [ks + 4·kb·sin²(q/2)] |z_q|²
    所以每个珠子的方差 = (1/N) Σ_q kt / [ks + 4·kb·sin²(q/2)]

    注意：√(kt/ks) = 0.45 nm 是"忽略弯曲"时的上界，
    真实的单珠涨落要小得多，因为短波模式被 kb 压住了。
    """
    q = 2 * np.pi * np.arange(n) / n
    stiff = ks + 4 * kb * np.sin(q / 2) ** 2
    return float(np.sqrt(np.mean(kt / stiff)))


sig_meas = float(np.mean(np.std(far_samp, axis=0))) if len(far_samp) else float("nan")
sig_theory = bead_std_theory()

# 诊断二：收敛性。后半段切成 NBLK 块，各自平均后与总平均的最大偏差。
blk_dev = float(np.max(np.abs(blk_mean - z_u_mean)[:, near_mask]))

# 诊断三：配体链的状态。首尾距比伸直长度短多少，就是被压弯了多少。
if N_LIG:
    _e2e = [float(np.linalg.norm(lig_fin[k * L_LIG] - lig_fin[k * L_LIG + L_LIG - 1]))
            for k in range(N_LIG)]
    lig_end_to_end = float(np.mean(_e2e))
else:
    lig_end_to_end = 0.0

if final_gap > 3.0:
    verdict = "颗粒脱附，飘到顶盖去了（没有吸附住）"
elif final_gap < -2.0:
    verdict = "颗粒钻进了膜内部"
elif dimple > 0.15:
    verdict = "颗粒被吸住，在膜上压出一个浅坑"
elif final_gap < 0.8:
    verdict = "颗粒贴在膜上，但膜几乎没怎么变形"
else:
    verdict = "颗粒只是碰了一下就离开，没有吸附住"

lines = [
    "二维布朗动力学：颗粒在膜上的吸附",
    "-" * 46,
    f"参数  N={N_COL}  A={A}  D0={D0}  R={R}  EPS={EPS}  K_S={K_S}  W={W}  kT={KT}",
    f"步数  {STEPS}  dt={DT}  总时间 {STEPS * DT:.0f}",
    "",
    f"初始间隙                {GAP0:+.2f} nm",
    f"接触时刻（间隙<0.3nm）  {('%.1f' % contact_time) if contact_time else '未发生'}",
    f"最终间隙                {final_gap:+.2f} nm",
    f"最大接触珠子数          {max_nc}",
    f"膜的静态凹陷深度        {dimple:.2f} nm",
    f"周围隆起高度            {ridge:.2f} nm",
    f"单珠热涨落 实测/理论    {sig_meas:.2f} / {sig_theory:.2f} nm",
    f"分块收敛性 最大偏差      {blk_dev:.3f} nm",
    f"颗粒步长被截断的比例    {clip_frac * 100:.2f} %（>1% 说明 STEP_MAX 太小，会压低有效温度）",
    f"颗粒撞到顶盖的比例      {ceil_frac * 100:.3f} %（>0 说明它脱附、飘走了）",
    f"下叶是否被吸引          {'是（--twoleaf，旧做法）' if ARGS.twoleaf else '否（默认：只近侧叶受力）'}",
    f"配体链                  {N_LIG} 条 × {L_LIG} 珠" if N_LIG else "配体链                  无（光板球）",
    f"贴膜配体珠数(距<0.9nm)  {lig_contact_mean:.1f} 个" if N_LIG else "",
    f"配体链平均首尾距        {lig_end_to_end:.2f} nm（伸直 {L_BOND * L_LIG:.2f}）" if N_LIG else "",
    f"结论                    {verdict}",
]
lines = [x for x in lines if x]
log = "\n".join(lines)
print(log)

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "sim_log.txt"), "w", encoding="utf-8") as f:
    f.write(log + "\n")

if ARGS.noplot:
    raise SystemExit(0)


# ----------------------------------------------------------------------
# 6. 画图（只用 PIL，不依赖 matplotlib）
# ----------------------------------------------------------------------
FONT_DIR = r"C:\Windows\Fonts"


def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    path = os.path.join(FONT_DIR, name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


INK = (28, 30, 34)
GREY = (120, 124, 132)
LIGHT = (186, 192, 202)
BLUE = (44, 92, 178)
PALE = (158, 184, 224)
RED = (196, 72, 60)
GREEN = (32, 122, 88)
BG = (255, 255, 255)

W_IMG, H_IMG = 1240, 1000
img = Image.new("RGB", (W_IMG, H_IMG), BG)
dr = ImageDraw.Draw(img)

# ---- 标题 ----
dr.text((70, 40), "纳米颗粒在脂质膜上的吸附", font=font(32, True), fill=INK)
dr.text((72, 88),
        f"二维布朗动力学 · 膜厚 {D0} nm · 颗粒直径 {2*R:.1f} nm · "
        f"吸引强度 EPS={EPS} · 温度 kT={KT} · 仿真 {STEPS*DT:.0f} 个约化时间单位",
        font=font(16), fill=GREY)

# ---- 面板 A：间隙随时间 ----
ax0, ax1, ay0, ay1 = 112, 1150, 168, 540
gmin = min(-1.0, float(np.min(gap_arr)) - 0.3)
gmax = max(GAP0 + 0.3, 0.6)


def AX(t):
    return ax0 + (t - t_arr[0]) / (t_arr[-1] - t_arr[0]) * (ax1 - ax0)


def AY(g):
    return ay1 - (g - gmin) / (gmax - gmin) * (ay1 - ay0)


dr.text((70, 136), "颗粒底面到远处膜面的间隙（nm）", font=font(15), fill=GREY)

gstep = 1 if gmax - gmin <= 6 else 2
g = np.ceil(gmin / gstep) * gstep
while g <= gmax:
    dr.line([(ax0, AY(g)), (ax1, AY(g))], fill=(240, 242, 246))
    dr.text((ax0 - 12, AY(g)), f"{g:g}", font=font(14), fill=GREY, anchor="rm")
    g += gstep

dr.line([(ax0, AY(0)), (ax1, AY(0))], fill=(214, 218, 226))
dr.line([(ax0, ay1), (ax1, ay1)], fill=LIGHT)

# 原始轨迹（浅色细线）+ 滑动平均（深色粗线）
dr.line([(AX(t), AY(g)) for t, g in zip(t_arr, gap_arr)], fill=PALE, width=1)

win = 40
kernel = np.ones(win) / win
smooth = np.convolve(gap_arr, kernel, mode="same")
smooth[: win // 2] = gap_arr[: win // 2]
smooth[-(win // 2):] = gap_arr[-(win // 2):]
dr.line([(AX(t), AY(g)) for t, g in zip(t_arr, smooth)], fill=BLUE, width=3)

# 接触时刻标记
if contact_time is not None and contact_time > 0.004 * t_arr[-1]:
    cx = AX(contact_time)
    dr.line([(cx, AY(0.3)), (cx, ay1)], fill=RED, width=1)
    dr.text((cx + 8, AY(0.3) - 22), f"t = {contact_time:.1f} 接触", font=font(14), fill=RED)

dr.text(((ax0 + ax1) / 2, ay1 + 30), "时间", font=font(15), fill=GREY, anchor="mt")

# 图例（放在面板右上角，避开纵轴标题）
dr.line([(ax1 - 250, ay0 - 14), (ax1 - 210, ay0 - 14)], fill=PALE, width=2)
dr.text((ax1 - 204, ay0 - 24), "原始轨迹", font=font(14), fill=GREY)
dr.line([(ax1 - 110, ay0 - 14), (ax1 - 70, ay0 - 14)], fill=BLUE, width=3)
dr.text((ax1 - 64, ay0 - 24), "滑动平均", font=font(14), fill=GREY)

# ---- 面板 B：放大到颗粒附近的正视图（画在子图上，自动裁剪）----
bx0, by0 = 110, 650                             # 面板左上角
half = 8.0                                      # 横向看 ±8 nm
bw = 520
ppn = bw / (2 * half)                           # 每 nm 多少像素（x/z 同一比例）
dmin, dmax = -2.6, 4.8                          # 纵向显示的高度范围（相对远处膜面）
bh = int(round((dmax - dmin) * ppn))

bimg = Image.new("RGB", (bw, bh), BG)
bdr = ImageDraw.Draw(bimg)


def BX(x):
    return (x - (PX - half)) * ppn


def BY(d):
    """d = 相对远处膜面的高度（nm），向上为正。"""
    return bh - (d - dmin) * ppn


# 自由膜面的参考线（远处膜面所在的水平线）
dash_y = BY(0.0)
for xd in range(0, bw, 16):
    bdr.line([(xd, dash_y), (xd + 8, dash_y)], fill=(226, 228, 234), width=1)

# 下叶（终态，浅灰）
bdr.line([(BX(x), BY(-D0 + s)) for x, s in zip(x_grid, shape * 0.35)], fill=LIGHT, width=2)
# 上叶：初始（水平，灰）与时间平均后的终态（蓝）
bdr.line([(BX(x), BY(0.0)) for x in x_grid], fill=(228, 230, 236), width=2)
bdr.line([(BX(x), BY(s)) for x, s in zip(x_grid, shape)], fill=BLUE, width=3)

# 颗粒（x/z 同比例，所以是真圆）
cx = BX(px)
cy = BY(final_gap + R)                          # 颗粒中心高出自由膜面的距离
rr = R * ppn
bdr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
            fill=(234, 242, 238), outline=GREEN, width=3)

# 配体链：从锚点连到珠子，珠子画成小圆（只画"占地方"的几何，不画势）
if N_LIG:
    rl_px = max(3, int(round(R_LIG * ppn * 0.8)))
    for k in range(N_LIG):
        pts = [(BX(px + ROOT[k][0]), BY(final_gap + R + ROOT[k][1]))]
        for i in range(L_LIG):
            j = k * L_LIG + i
            pts.append((BX(px + lig_fin[j, 0]), BY(final_gap + R + lig_fin[j, 1])))
        bdr.line(pts, fill=(196, 148, 72), width=2)
        for xp, yp in pts[1:]:
            bdr.ellipse([xp - rl_px, yp - rl_px, xp + rl_px, yp + rl_px],
                        fill=(246, 214, 160), outline=(176, 128, 52), width=1)

img.paste(bimg, (bx0, by0))

dr.text((70, 614), "颗粒附近的正视图（横向 ±8 nm，纵横同比例）", font=font(15), fill=GREY)
dr.text((70, by0 + bh + 14),
        "蓝线＝膜的时间平均形状　灰线＝未受扰动的平膜　绿圆＝颗粒　黄珠＝配体　虚线＝远处膜面",
        font=font(14), fill=GREY)

# ---- 面板 C：读数 ----
tx = 700
dr.text((tx, 614), "读数", font=font(15), fill=GREY)

rows = [
    ("最终间隙", f"{final_gap:+.2f} nm"),
    ("膜上凹坑", f"{dimple:.2f} nm"),
    ("接触珠子", f"{max_nc} 个"),
]
y = by0 + 4
for k, v in rows:
    dr.text((tx, y), k, font=font(18), fill=INK)
    dr.text((tx + 160, y), v, font=font(18, True), fill=INK)
    y += 44

dr.line([(tx, y + 8), (tx + 460, y + 8)], fill=(232, 234, 240))
dr.text((tx, y + 22), "结论", font=font(15), fill=GREY)
dr.text((tx, y + 52), verdict, font=font(17, True),
        fill=GREEN if dimple > 0.15 else RED)

dr.text((tx, y + 104), "只改一个数，看结局怎么变：", font=font(15), fill=GREY)
dr.text((tx, y + 132), "python sim.py --eps=2", font=font(14), fill=INK)
dr.text((tx + 220, y + 132), "只贴上，压不出坑", font=font(14), fill=GREY)
dr.text((tx, y + 156), "python sim.py --eps=120", font=font(14), fill=INK)
dr.text((tx + 220, y + 156), "吸得越来越深", font=font(14), fill=GREY)

out = os.path.join(here, "fig_adsorption.png")
img.save(out)
print("figure saved:", out)


# ----------------------------------------------------------------------
# 7. 附加图：收敛性检查（后半段分块，各自的平均形状是否重合）
# ----------------------------------------------------------------------
def free_plane(z):
    """扣掉远处膜面的整体倾斜，得到相对自由膜面的形状。"""
    if far_mask.sum() > 5:
        c = np.polyfit(x_grid[far_mask], z[far_mask], 1)
        return z - np.polyval(c, x_grid)
    return z - float(np.median(z))


CW, CH = 1240, 600
cimg = Image.new("RGB", (CW, CH), BG)
cdr = ImageDraw.Draw(cimg)
cdr.text((70, 34), "收敛性检查：后半段切成 4 块，各自的平均膜形状", font=font(28, True), fill=INK)
cdr.text((72, 78),
         f"K_S={K_S}　EPS={EPS}　每块约 {0.4 * STEPS * DT / ARGS.blocks:.0f} 个时间单位　"
         f"块间最大偏差 {blk_dev:.3f} nm",
         font=font(15), fill=GREY)

kx0, kx1, ky0, ky1 = 110, 1160, 150, 470
khalf = 8.0
kmin, kmax = -1.0, 0.6


def KX(x):
    return kx0 + (x - (PX - khalf)) / (2 * khalf) * (kx1 - kx0)


def KY(v):
    return ky1 - (v - kmin) / (kmax - kmin) * (ky1 - ky0)


for v in (0.0, -0.5, 0.5):
    cdr.line([(kx0, KY(v)), (kx1, KY(v))], fill=(240, 242, 246))
    cdr.text((kx0 - 12, KY(v)), f"{v:+.1f}", font=font(14), fill=GREY, anchor="rm")
cdr.line([(kx0, KY(0)), (kx1, KY(0))], fill=(214, 218, 226))

BLK_COLORS = [(44, 92, 178), (32, 122, 88), (196, 72, 60), (196, 150, 40)]
for k in range(ARGS.blocks):
    bshape = free_plane(blk_mean[k])
    cdr.line([(KX(x), KY(s)) for x, s in zip(x_grid, bshape)],
             fill=BLK_COLORS[k % 4], width=2)
cdr.line([(KX(x), KY(s)) for x, s in zip(x_grid, shape)], fill=INK, width=3)

cdr.text((kx0, ky1 + 26), "x（nm，以颗粒为中心）", font=font(15), fill=GREY)
for k in range(ARGS.blocks):
    lx = kx1 - 420 + (k % 2) * 210
    ly = ky0 - 34 + (k // 2) * 22
    cdr.line([(lx, ly), (lx + 26, ly)], fill=BLK_COLORS[k % 4], width=2)
    cdr.text((lx + 32, ly - 9), f"第 {k + 1} 块", font=font(13), fill=GREY)
cdr.text((70, ky1 + 60),
         "四条彩线几乎重合＝你的凹陷深度 是收敛的；分叉说明跑得还不够久。"
         "黑线＝全部后半段的平均。",
         font=font(14), fill=GREY)

cout = os.path.join(here, "fig_convergence.png")
cimg.save(cout)
print("figure saved:", cout)
