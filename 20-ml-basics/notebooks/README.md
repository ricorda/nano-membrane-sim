# Notebook 规范

每个 notebook 一个主题，文件名 `NN-主题.ipynb`（如 `01-梯度下降.ipynb`）。

要求：

1. **第一格是 markdown**：写清这个 notebook 要回答什么问题。
2. **最后一格是结论**：用两三句话写"我看到了什么"，不要只有图没有话。
3. 图必须有标题、坐标轴标签、图例。
4. 随机过程固定 `np.random.seed(...)`，保证可复现。
5. 跑完清空大输出再提交（`jupyter nbconvert --clear-output --inplace *.ipynb` 或手动清）。
