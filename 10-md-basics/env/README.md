# 环境搭建记录

> 目的：把"我是怎么装上的"记下来。将来换机器、或者帮别人装，不用重新踩坑。

## 方案 A：WSL2 + conda（推荐，Windows 上最省事）

```bash
# 1. 管理员 PowerShell 里装 WSL2（只需一次，装完重启）
wsl --install -d Ubuntu

# 2. 进入 Ubuntu 后装 miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 建环境
conda create -n md python=3.11 -y
conda activate md
conda install -c conda-forge openmm mdtraj nglview -y
pip install matplotlib numpy pandas
```

验证：

```bash
python -c "import openmm; print(openmm.__version__)"
```

## 方案 B：Windows 原生

- GROMACS 有 Windows 版，但教程和报错信息几乎都是 Linux 的，遇到问题不好搜。
- OpenMM 可以纯 Python 装（`conda install -c conda-forge openmm`），能跑，但性能不如 Linux。

**结论：能上 WSL2 就上 WSL2。**

## 记录表

| 日期 | 做了什么 | 结果 | 踩的坑 |
|---|---|---|---|
| | | | |

## 常用命令备忘

（把自己实际用到的命令贴在这里，别抄网上的）
