# %% [markdown]
# # Day 54：层次聚类
#
# 目标：用树状图理解逐步合并，并比较不同簇间距离。凝聚法从每个样本一个簇开始；切树高度或指定簇数决定最终分组。
#
# 运行前请阅读[环境与运行说明](../docs/setup.md)。本课 `.py` 是教学源文件，配套 Markdown 和 Notebook 自动同步。图形保存到 `outputs/`，设置 `COURSE_SHOW_PLOTS=1` 可显示窗口。

# %%
from pathlib import Path
import sys

# 脚本从文件位置定位仓库；Notebook 从当前工作目录向上查找。
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for candidate in (base, *base.parents):
    if (candidate / "Code" / "course_utils.py").is_file():
        code_dir = str(candidate / "Code")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        break
else:
    raise FileNotFoundError("找不到课程仓库，请从仓库根目录或 Code 目录启动 Notebook。")
from course_utils import DATA, OUTPUT, finish_plot


# %% [markdown]
# ## 标准化与树状图
#
# single 取最近点距离，complete 取最远点距离，average 取平均距离，Ward 选择使簇内平方和增加最小的合并。Ward 要求欧氏距离。树状图适合较小数据集，大样本会有时间和内存成本。linkage 返回 (n−1,4) 合并表，每行记录两个簇编号、合并距离、合并后的样本数；Ward 的图示距离与平方和增量相关，但不是直接等于平方和增量。

# %%
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.7, random_state=0)
scaled = StandardScaler().fit_transform(X)
Z = linkage(scaled, method="ward", metric="euclidean")
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=12)
ax.set(xlabel="Sample/group", ylabel="Ward linkage distance")
finish_plot("day54_dendrogram")
for method in ["single", "complete", "average", "ward"]:
    model = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage=method)
    labels = model.fit_predict(scaled)
    print(method, "silhouette:", silhouette_score(scaled, labels))

# %% [markdown]
# ## 练习与检查
#
# 比较加入离群点后 single 的链式效应与 complete 的结果。尝试 `n_clusters=None, distance_threshold=...` 按距离切树。该估计器没有对新样本直接 `predict` 的接口，不要把它当监督分类器使用。
