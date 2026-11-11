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
          ② 高斯吸引壳 EPS  —— 距离超过 R 之后按 exp(-(d-R)/W) 衰减，
             超过 R+3W 截断。这就是"黏性"的来源。

运动  = 过阻尼朗之万：每一步  x += F*dt + sqrt(2*dt)*xi
        没有质量、没有惯性，力大就流得快。xi 是标准差为 1 的高斯白噪声。
        温度取 kT = 1（约化单位）。

运行
----
    python sim.py                    # 默认参数，出图
    python sim.py --eps=5            # 把吸引强度调大
    python sim.py --eps=0.15         # 几乎不吸引，看它会不会弹开
    python sim.py --steps=200000     # 跑更久
"""

import argparse
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
    p.add_argument("--gap", type=float, default=3.0, help="颗粒初始间隙 nm，默认 3.0")
    p.add_argument("--ks", type=float, default=5.0, help="膜的软硬（拴在晶格点的弹簧刚度），默认 5")
    p.add_argument("--seed", type=int, default=1, help="随机种子")
    p.add_argument("--noplot", action="store_true", help="只算不画图")
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
STEP_MAX = 0.05     # 单步最大位移，防止爆炸
P_MOB = 0.6         # 颗粒的迁移率因子（颗粒比珠子大，动得慢）
REC = 20            # 每隔多少步记录一次

L = N_COL * A       # 膜的总长度 nm
PX = L / 2          # 颗粒的 x 位置（膜中央，固定不动）

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
      R <= s < R+CUT : 高斯吸引，把珠子往颗粒拉，颗粒被往膜拉
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

    for dist, dz, f_out in ((dist_u, dz_u, f_u), (dist_d, dz_d, f_d)):
        safe = np.maximum(dist, 1e-9)
        ux = dx / safe                 # 从颗粒指向珠子的单位矢量
        uz = dz / safe

        # ① 硬核排斥
        inside = dist < R
        if np.any(inside):
            mag = K_CORE * (R - dist[inside])
            f_out[inside] += mag * ux[inside]
            f_out[inside] += mag * uz[inside]
            fp_z += -np.sum(mag * uz[inside])

        # ② 高斯吸引
        shell = (dist >= R) & (dist < R + CUT)
        if np.any(shell):
            mag = EPS * np.exp(-(dist[shell] - R) / W)
            f_out[shell] += -mag * ux[shell]
            f_out[shell] += -mag * uz[shell]
            fp_z += np.sum(mag * uz[shell])
            n_contact += int(np.sum(dist[shell] < R + 0.35))

    return f_u, f_d, fp_z, n_contact


# ----------------------------------------------------------------------
# 4. 主循环
# ----------------------------------------------------------------------
def run():
    global z_u, z_d, pz

    hist_t, hist_gap, hist_nc, hist_bump = [], [], [], []
    prof_u = np.zeros(N_COL)       # 后半段的时间平均形状，压掉热噪声
    prof_n = 0

    for step in range(STEPS + 1):
        # ---- 膜自身的力 ----
        fu, fd = membrane_forces(z_u, z_d)

        # ---- 颗粒与膜之间的力 ----
        dfu, dfd, fp_z, n_contact = particle_forces(z_u, z_d, pz)
        fu += dfu
        fd += dfd

        # ---- 积分（过阻尼朗之万）----
        z_u += DT * fu + np.sqrt(2 * DT) * rng.standard_normal(N_COL)
        z_d += DT * fd + np.sqrt(2 * DT) * rng.standard_normal(N_COL)

        dp = P_MOB * (DT * fp_z + np.sqrt(2 * DT) * rng.standard_normal())
        dp = np.clip(dp, -STEP_MAX, STEP_MAX)
        pz += dp

        # ---- 记录 ----
        if step % REC == 0:
            t = step * DT
            far = np.abs(x_grid - px) > 4.0
            base = float(np.median(z_u[far])) if far.sum() > 5 else float(np.median(z_u))
            gap = (pz - R) - base          # 颗粒底面到远处膜面的距离
            near = np.abs(x_grid - px) <= 4.0
            bump = float(np.max(z_u[near]) - base) if near.any() else 0.0
            hist_t.append(t)
            hist_gap.append(gap)
            hist_nc.append(n_contact)
            hist_bump.append(bump)

            # 只对后半段（已经吸附住之后）做时间平均
            if t > 0.6 * STEPS * DT:
                prof_u += z_u
                prof_n += 1

    return (np.array(hist_t), np.array(hist_gap),
            np.array(hist_nc), np.array(hist_bump),
            prof_u / max(prof_n, 1), z_u.copy(), pz)


t_arr, gap_arr, nc_arr, bump_arr, z_u_mean, z_u_fin, pz_fin = run()

# 时间平均形状：先扣掉远处膜面的整体倾斜，再相对远处高度归零
far_mask = np.abs(x_grid - px) > 4.0
if far_mask.sum() > 5:
    coef = np.polyfit(x_grid[far_mask], z_u_mean[far_mask], 1)
    plane = np.polyval(coef, x_grid)
else:
    plane = np.full(N_COL, float(np.median(z_u_mean)))
shape = z_u_mean - plane

# ----------------------------------------------------------------------
# 5. 判读结果
# ----------------------------------------------------------------------
final_gap = float(np.mean(gap_arr[-50:]))          # 末尾取平均，压掉热涨落
max_bump = float(np.max(bump_arr))
max_nc = int(np.max(nc_arr))
contact_idx = np.where(gap_arr < 0.3)[0]
contact_time = float(t_arr[contact_idx[0]]) if len(contact_idx) else None

# 静态形变：看时间平均后的膜形状，而不是某一帧的瞬时值
near_mask = np.abs(x_grid - px) <= 2.5
dimple = float(-shape[near_mask].min())            # 颗粒正下方被压下去的深度
ridge = float(shape[near_mask].max())              # 周围被拉起来的高度

if final_gap < -2.0:
    verdict = "颗粒钻进了膜内部"
elif dimple > 0.25:
    verdict = "颗粒被吸住，在膜上压出一个可见的坑"
elif final_gap < 0.8:
    verdict = "颗粒贴在膜上，但膜几乎没怎么变形"
else:
    verdict = "颗粒只是碰了一下就离开，没有吸附住"

lines = [
    "二维布朗动力学：颗粒在膜上的吸附",
    "-" * 46,
    f"参数  N={N_COL}  A={A}  D0={D0}  R={R}  EPS={EPS}  K_S={K_S}  W={W}",
    f"步数  {STEPS}  dt={DT}  总时间 {STEPS * DT:.0f}",
    "",
    f"初始间隙                {GAP0:+.2f} nm",
    f"接触时刻（间隙<0.3nm）  {('%.1f' % contact_time) if contact_time else '未发生'}",
    f"最终间隙                {final_gap:+.2f} nm",
    f"最大接触珠子数          {max_nc}",
    f"膜的静态凹陷深度        {dimple:.2f} nm",
    f"周围隆起高度            {ridge:.2f} nm",
    f"结论                    {verdict}",
]
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
        f"吸引强度 EPS={EPS} · 仿真 {STEPS*DT:.0f} 个约化时间单位",
        font=font(16), fill=GREY)

# ---- 面板 A：间隙随时间 ----
ax0, ax1, ay0, ay1 = 112, 1150, 168, 540
gmin, gmax = -1.0, max(3.2, GAP0 + 0.3)


def AX(t):
    return ax0 + (t - t_arr[0]) / (t_arr[-1] - t_arr[0]) * (ax1 - ax0)


def AY(g):
    return ay1 - (g - gmin) / (gmax - gmin) * (ay1 - ay0)


dr.text((70, 136), "颗粒底面到远处膜面的间隙（nm）", font=font(15), fill=GREY)

for g in (0, 1, 2, 3):
    if gmin <= g <= gmax:
        dr.line([(ax0, AY(g)), (ax1, AY(g))], fill=(240, 242, 246))
        dr.text((ax0 - 12, AY(g)), f"{g}", font=font(14), fill=GREY, anchor="rm")

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
dmin, dmax = -2.6, 3.4                          # 纵向显示的高度范围（相对远处膜面）
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

img.paste(bimg, (bx0, by0))

dr.text((70, 614), "颗粒附近的正视图（横向 ±8 nm，纵横同比例）", font=font(15), fill=GREY)
dr.text((70, by0 + bh + 14),
        "蓝线＝膜的时间平均形状　灰线＝未受扰动的平膜　绿圆＝颗粒　虚线＝远处膜面",
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
        fill=GREEN if dimple > 0.25 else RED)

dr.text((tx, y + 104), "只改一个数，看结局怎么变：", font=font(15), fill=GREY)
dr.text((tx, y + 132), "python sim.py --eps=2", font=font(14), fill=INK)
dr.text((tx + 220, y + 132), "只贴上，压不出坑", font=font(14), fill=GREY)
dr.text((tx, y + 156), "python sim.py --eps=120", font=font(14), fill=INK)
dr.text((tx + 220, y + 156), "吸得越来越深", font=font(14), fill=GREY)

out = os.path.join(here, "fig_adsorption.png")
img.save(out)
print("figure saved:", out)
