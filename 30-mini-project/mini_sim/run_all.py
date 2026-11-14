"""run_all.py —— 一条命令复现仓库里所有的图和数。

为什么要有这个文件：
    别人（和你自己三个月后）拿到仓库，最想知道的是"从哪里开始"。
    有这一个入口，就不用猜 sim.py / scan.py / gp.py 的先后顺序。

用法
----
    python run_all.py            # 主图：默认构型 + 覆盖率对照（约 1 分钟）
    python run_all.py --full     # 再加 25 点相图 + 主动学习对照（约 25 分钟）
    python run_all.py --fresh    # 忽略缓存，全部重算

产出（都在本目录）
------------------
    fig_adsorption.png   主图：间隙曲线 + 构型 + 读数
    fig_convergence.png  收敛性：后半段四块的平均膜形状
    fig_cooling.png      降温判据（读 _kt_cache.json）
    fig_coverage.png     覆盖率对照（读 _cov.json）
    fig_phase.png        --full 时才有：相图
    fig_al.png           --full 时才有：主动学习对照
    sim_log.txt          最后一次默认运行的日志
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
FRESH = "--fresh" in sys.argv
FULL = "--full" in sys.argv


def step(title, cmd):
    print(f"\n=== {title} ===")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE)
    if p.returncode != 0:
        print(f"!!! 失败（退出码 {p.returncode}）：{' '.join(cmd)}")
        raise SystemExit(1)
    print(f"    完成，{time.time() - t0:.0f} 秒")


def main():
    step("默认构型（主图 + 收敛性 + 日志）", [PY, "sim.py"])
    step("降温判据", [PY, "sweep_kt.py"] + (["--fresh"] if FRESH else []))
    step("覆盖率对照", [PY, "coverage.py"] + (["--fresh"] if FRESH else []))

    if FULL:
        step("25 点扫描（约 20 分钟）", [PY, "scan.py"])
        step("相图：9 点拟合", [PY, "gp.py"])
        step("主动学习 vs 随机", [PY, "al.py"])

    print("\n全部完成。图在：fig_adsorption / fig_convergence / fig_cooling / fig_coverage"
          + (" / fig_phase / fig_al" if FULL else ""))


if __name__ == "__main__":
    main()
