# 外部教材与现代 API

Day 45～53 的旧《Python 数据科学手册》链接保留在[历史路线](original-roadmap.md)。当前学习优先使用[本地逐日讲解](curriculum.md)，旧教材可帮助理解概念，但其环境和网络依赖不作为本课程运行保证。

| 旧写法/材料 | 当前处理 |
|---|---|
| `sklearn.cross_validation` | `sklearn.model_selection` |
| `sklearn.preprocessing.Imputer` | `sklearn.impute.SimpleImputer` |
| `OneHotEncoder(categorical_features=...)` | `ColumnTransformer` 选择列，再使用 `OneHotEncoder` |
| `get_feature_names()` | 相关转换器的 `get_feature_names_out()` |
| `OneHotEncoder(sparse=False)` | 新版使用 `sparse_output=False` |
| `df1.append(df2)` | `pd.concat([df1, df2])` |
| `plt.style.use('seaborn-whitegrid')` | `seaborn-v0_8-whitegrid`，或直接不用样式 |
| `plt.style.use('seaborn-white')` | `seaborn-v0_8-white`，或直接不用样式 |
| 时间频率 `BA` | 按目标版本使用明确别名，如 `BYE` |
| `model.save('name.model')` | 完整 Keras 模型用 `.keras`；SavedModel 部署用 `export` |
| 猫狗数据全量 pickle | JSON 清单 + 批次解码；保留划分与哈希 |

这些替换解决接口问题，不自动保证旧程序的统计方法正确。例如测试集仍不能参与填补/缩放拟合，模型选择不能反复查看测试成绩。

## 参考入口

- [NumPy 学习指南](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [pandas 入门教程](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
- [Matplotlib 官方教程](https://matplotlib.org/stable/tutorials/index.html)
- [scikit-learn 常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html)
- [Keras 3 迁移](https://keras.io/guides/migrating_to_keras_3/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

书籍、PDF 速查表和历史视频没有逐页/逐帧核查；请核对出版时间和适用版本。本轮只修改代码和文字材料，未编辑图片。
