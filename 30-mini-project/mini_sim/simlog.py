"""simlog.py —— 统一解析 sim.py 打印到 stdout 的那几行数。

原来 scan.py / sweep_kt.py / coverage.py 各写了一份正则，改一个日志格式就得改三处。
放到这里，谁要读数就 import 谁。

用法：
    from simlog import parse
    d = parse(out)          # d["gap"], d["dimple"]；没找到的键是 None
"""

import re

_GAP = re.compile(r"最终间隙\s+([-+]?\d+\.\d+)")
_DIM = re.compile(r"膜的静态凹陷深度\s+([-+]?\d+\.\d+)")
_BLK = re.compile(r"分块收敛性 最大偏差\s+([-+]?\d+\.\d+)")
_CLIP = re.compile(r"颗粒步长被截断的比例\s+([-+]?\d+\.\d+)")
_CONTACT = re.compile(r"接触珠子\s+(\d+) 个")
_LIG = re.compile(r"配体链\s+(\d+) 条 × (\d+) 珠")


def _one(pat, text, cast=float):
    m = pat.search(text)
    return cast(m.group(1)) if m else None


def parse(text):
    """把 sim.py 的输出变成字典。缺失的项为 None，不抛异常。"""
    out = {
        "gap": _one(_GAP, text),
        "dimple": _one(_DIM, text),
        "blk": _one(_BLK, text),
        "clip": _one(_CLIP, text),
        "contact": _one(_CONTACT, text, int),
    }
    m = _LIG.search(text)
    out["nlig"], out["beads_per_chain"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return out
