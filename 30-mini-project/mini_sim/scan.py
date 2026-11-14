"""scan.py —— 把参数空间扫成一张表，存成 JSON。

用法
----
    python scan.py                # 扫默认的 5×5 网格，约 5 分钟
    python scan.py --quick        # 只扫 9 点，调试用

两个轴
------
    配体数 nlig:  0, 4, 7, 10, 14      （0 = 光板球，14 = 二维几何上限）
    吸引强度 eps: 12, 22, 40, 70, 120  （12 几乎不吸，120 会压穿）

响应量
------
    y = -gap  （nm），越大表示颗粒陷得越深。
    用 -gap 而不是 gap，是因为"吸附深度"这个说法更直观：
    0 表示刚贴上，0.9 表示已经快到膜中间。

为什么存 JSON 而不是 CSV
------------------------
    点太少（25 个），JSON 能顺手把参数、时间戳、版本号一起存进去，
    以后换模型重扫时可以对比。JSON 也只有几十行，人眼能读。
"""

import argparse
import json
import os
import subprocess
import sys
import time

from simlog import parse

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim.py")
OUT = os.path.join(HERE, "_scan.json")

NL = [0, 4, 7, 10, 14]
EP = [12, 22, 40, 70, 120]

QUICK_NL = [0, 7, 14]
QUICK_EP = [12, 40, 120]


def run_one(nlig, eps, seed=1, steps=120000):
    cmd = [
        sys.executable, SIM,
        f"--nlig={nlig}", f"--eps={eps}",
        f"--seed={seed}", f"--steps={steps}", "--noplot",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"nlig={nlig} eps={eps} 失败：\n{p.stdout}\n{p.stderr}")
    d = parse(p.stdout)
    if d["gap"] is None or d["dimple"] is None:
        raise RuntimeError(f"日志解析失败：\n{p.stdout}")
    return {
        "nlig": nlig,
        "eps": eps,
        "gap": d["gap"],
        "dimple": d["dimple"],
        "blk": d["blk"],
        "sec": round(dt, 1),
    }


def replicate_boundary(rows, seeds=(2, 3), steps=120000):
    """挑出"可疑点"，多跑种子取中位数。

    挑法不是按 gap 的绝对值，而是按**不连续性**：
    同一个配体数下，把某个点跟左右两个 eps 邻居比，如果它比两边平均
    高出或低出 0.45 nm 以上，说明这里很可能跨过了吸附/脱附边界
    （双态点：同一组参数可能吸住也可能飘走）。远离边界的点种子间
    极差 < 0.05 nm，补种子纯属浪费。
    """
    by_key = {(r["nlig"], r["eps"]): r for r in rows}
    eps_list = sorted({r["eps"] for r in rows})
    suspect = []
    for r in rows:
        i = eps_list.index(r["eps"])
        nb = [by_key[(r["nlig"], eps_list[j])] for j in (i - 1, i + 1)
              if 0 <= j < len(eps_list) and (r["nlig"], eps_list[j]) in by_key]
        if len(nb) < 2:
            continue
        ref = sum(n["gap"] for n in nb) / 2
        if abs(r["gap"] - ref) > 0.45:
            suspect.append((r["nlig"], r["eps"]))

    if not suspect:
        print("    没有发现可疑点")
        return rows

    print(f"    可疑点 {len(suspect)} 个：{suspect}")
    out = []
    for r in rows:
        if (r["nlig"], r["eps"]) not in suspect:
            out.append(r)
            continue
        gaps = [r["gap"]]
        for sd in seeds:
            rr = run_one(r["nlig"], r["eps"], seed=sd, steps=steps)
            gaps.append(rr["gap"])
            print(f"      补种子 nlig={r['nlig']} eps={r['eps']} seed={sd} "
                  f"gap={rr['gap']:+.2f}", flush=True)
        gaps.sort()
        med = gaps[len(gaps) // 2]
        print(f"      → {['%+.2f' % g for g in gaps]} 取中位数 {med:+.2f}", flush=True)
        r = dict(r)
        r["gap"] = med
        r["replicated"] = [round(g, 2) for g in gaps]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只扫 3×3 的九点")
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-replicate", action="store_true",
                    help="边界点不补种子（默认会补，因为边界点是双态的）")
    args = ap.parse_args()

    nl_list = QUICK_NL if args.quick else NL
    ep_list = QUICK_EP if args.quick else EP

    rows = []
    total = len(nl_list) * len(ep_list)
    k = 0
    for nl in nl_list:
        for ep in ep_list:
            k += 1
            r = run_one(nl, ep, seed=args.seed, steps=args.steps)
            rows.append(r)
            print(f"[{k:2d}/{total}] nlig={nl:2d} eps={ep:3d}  "
                  f"gap={r['gap']:+.2f}  dimple={r['dimple']:.2f}  ({r['sec']}s)",
                  flush=True)

    if not args.no_replicate:
        print("\n检查边界点（gap 落在吸附/脱附过渡带上的点，单次结果不可信）")
        rows = replicate_boundary(rows, steps=args.steps)

    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": args.steps,
        "seed": args.seed,
        "nlig": nl_list,
        "eps": ep_list,
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n写入 {OUT}：{len(rows)} 个点")


if __name__ == "__main__":
    main()
