# CSE258R Assignment 2｜Baseline 实验记录

**研究问题**：对已有评分历史的用户，能否根据食材及个人口味，预测其对尚无历史评分的菜谱会给几星？

baseline 部分回答两个前置问题：不做个性化、只看食材，能预测到什么程度？只用"这个用户一贯怎么打分"，又能做到什么程度？个性化模型（`personalization/`）需要在同一套样本和指标下超过这些 baseline，才能说明"食材 + 个人口味"真正带来了提升。

## 运行方式

打开 `baseline/baseline.ipynb`，从上到下 Run All，约 3 分钟。notebook 依次完成：读数据 → 定义评估函数和模型 → 在验证冷启动行上调参（附三张调参图）→ 验证结果 → 用 train + 全部 validation 按已选参数重训，输出测试预测，再读 `test_labels.csv` 评分。所有结果写入 `baseline/outputs/`。

依赖 numpy、pandas、scipy、scikit-learn、matplotlib。需要先按仓库根目录 `readme.md` 解压 `data_prepared/`。notebook 会从当前目录向上查找 `data_prepared/shared`，所以在仓库内任意位置打开都可以。

## 评估协议

| 阶段 | 训练数据 | 评估数据 | 用途 |
| --- | --- | --- | --- |
| 验证 | `train.csv`，748,910 条，2010-04-25 前 | `validation.csv` 中 `is_cold_start_eval=1` 的 24,252 行（3,695 个用户） | 选模型和超参数 |
| 测试 | `train.csv` + 全部 `validation.csv`，909,740 条 | `test_features.csv` 的 11,668 行（1,872 个用户），按 `row_id` 合并 `test_labels.csv` | 参数已固定，只做最终评分 |

- **指标**：MAE 和 MSE 都汇报。
- **95% CI**：按**用户**有放回重抽样 1000 次，每次重新计算指标，取 2.5%–97.5% 分位。按用户而不是按行抽样，因为同一用户的评分相互相关，并且 73% 的评估行来自训练期评分 ≥100 条的重度用户。比较两个模型时，应看配对 bootstrap 差值的 CI，不能只看两个区间是否重叠。
- **分组**：测试集按时间分为开始后第一年（2012-09-11 至 2013-09-10）和之后，并与整体放在同一张表中对照。
- **防泄漏**：用户统计量只来自对应阶段的训练数据，在截止日冻结。TF-IDF 的词表和 IDF 只用训练菜谱计算。超参数只在验证集上选。

## 六个 baseline

| 模型 | 预测方法 | 原理 / 课程对应 | 调参 |
| --- | --- | --- | --- |
| `global_mean` | 训练集评分均值，4.67 | MSE 下最优的常数预测 | — |
| `global_median` | 训练集评分中位数，即全部预测 5 | MAE 下最优的常数预测。76% 的评分是 5 星 | — |
| `user_mean` | `(用户评分和 + k × 全局均值) / (用户评分数 + k)` | 偏置模型 `α + β_u + β_i` 在冷启动下的形态：新菜谱没有 `β_i`，只剩 `α + β_u`。`k` 等价于 `β_u` 的正则系数 λ，历史少的用户被拉向全局均值 | `k` 按验证 MSE 选，结果为 5 |
| `user_median` | 在用户历史中补 `k` 条 5 星后取中位数 | MAE 下最优的个性化常数。`k` 越大，越需要更多非 5 星证据才会预测低于 5 | `k` 按验证 MAE 选，结果为 2 |
| `ridge_tfidf` | 食材 TF-IDF → Ridge 回归 | 不做个性化的内容模型：所有用户共用一组食材系数（text mining + regression 两讲） | 按验证 MSE 选：按词切分，`min_df=5`，`alpha=300` |
| `ridge_tfidf_shifted` | `clip(Ridge 预测 + c, 1, 5)` | Ridge 拟合的是条件均值，MAE 需要的是中位数，所以用常数偏移 `c` 做校准 | `c` 按验证 MAE 选，结果为 +0.65 |

**TF-IDF 的设置**：每道菜谱的食材列表视为一篇文档。比较了两种切分方式：整个食材名作为一个词（`whole`，如 "chicken breast"），以及按空格拆成单词（`word`，如 "chicken" 和 "breast"，相关食材可以共享权重）。另外比较了 `min_df ∈ {2, 5, 20}` 和 `alpha ∈ {10, …, 1000}`，共 30 组。所有组合的验证 MSE 都在 0.369–0.372 之间，`word` 略好。

## 结果

### 总体

| 模型 | 验证 MAE | 验证 MSE | 测试 MAE | 测试 MAE 95% CI | 测试 MSE | 测试 MSE 95% CI |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `global_mean` | 0.4720 | 0.3744 | 0.4771 | [0.460, 0.496] | 0.4022 | [0.361, 0.451] |
| `global_median` | 0.2996 | 0.4633 | 0.3061 | [0.275, 0.339] | 0.4955 | [0.436, 0.563] |
| `user_mean` | 0.3895 | **0.3301** | 0.4026 | [0.375, 0.433] | **0.3536** | [0.317, 0.398] |
| `user_median` | **0.2842** | 0.4220 | **0.2935** | [0.264, 0.323] | 0.4446 | [0.395, 0.502] |
| `ridge_tfidf` | 0.4649 | 0.3695 | 0.4661 | [0.447, 0.486] | 0.3958 | [0.355, 0.445] |
| `ridge_tfidf_shifted` | 0.2996 | 0.4633 | 0.3061 | [0.275, 0.339] | 0.4955 | [0.436, 0.563] |

### 测试集：整体与按时间

| 模型 | MAE 整体 | MAE 第一年 | MAE 之后 | MSE 整体 | MSE 第一年 | MSE 之后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_mean` | 0.477 | 0.475 | 0.479 | 0.402 | 0.389 | 0.418 |
| `global_median` | 0.306 | 0.306 | 0.306 | 0.496 | 0.483 | 0.512 |
| `user_mean` | 0.403 | 0.398 | 0.408 | **0.354** | **0.338** | **0.374** |
| `user_median` | **0.294** | **0.282** | 0.308 | 0.445 | 0.416 | 0.480 |
| `ridge_tfidf` | 0.466 | 0.465 | 0.468 | 0.396 | 0.382 | 0.413 |
| 行数 | 11,668 | 6,449 | 5,219 | | | |

`ridge_tfidf_shifted` 与 `global_median` 完全相同，表中省略。

## 主要发现

1. **每个指标下胜出的，都是与其损失一致的模型。** MSE 下，均值类模型更好，`user_mean` 最优；MAE 下，中位数类模型更好，`user_median` 最优。`global_median` 在 MAE 下排第二，在 MSE 下却最差。两个指标会给出相反的排名，汇报时需要说明这一取舍。
2. **只看食材，信号很弱。** `ridge_tfidf` 相比 `global_mean`，测试 MSE 只从 0.402 降到 0.396，验证集上预测值的标准差只有 0.064。在 MAE 下做偏移校准后，最优偏移 +0.65 把所有预测都推到 5，`ridge_tfidf_shifted` 与 `global_median` 完全相同。也就是说，没有哪道菜的食材差到能让中位评分低于 5。
3. **用户历史会"过时"。** `user_median` 在测试第一年的 MAE 是 0.282，明显优于全部预测 5（0.306）；之后降到 0.308，已经不如全部预测 5。所有模型在之后时段的 MSE 也都更高。用户画像冻结在 2012-09-10，越往后越不准，可以联系 temporal 一讲的 drift 讨论。
4. **验证和测试的结论一致。** 六个模型在两个指标上的排名完全一样。测试分数整体略差（MAE 高 0.001–0.013，MSE 高 0.022–0.033），符合时间外推的预期。

**Ridge 食材系数**（`word` 切分，正值表示让评分升高）：

| 加分 | 系数 | 减分 | 系数 |
| --- | ---: | --- | ---: |
| butter | +0.145 | sprite | −0.245 |
| bacon | +0.143 | tofu | −0.236 |
| strawberries | +0.129 | applesauce | −0.229 |
| roast | +0.129 | macaroni | −0.218 |
| whipping | +0.126 | noodles | −0.205 |
| yeast | +0.122 | substitute | −0.196 |
| tortilla | +0.120 | rice | −0.189 |
| chipotle | +0.115 | chops | −0.188 |

减分项里有不少是"健康替代品"（substitute 多出现在 egg substitute、sugar substitute，以及 tofu、applesauce），另外还有 rice、pasta、noodles 等主食。这只是相关性，系数也很小。

## 输出文件

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `outputs/predictions/val_<model>.csv` | 24,252 行，`row_id,prediction` | 各模型的验证预测。和个性化模型做配对 bootstrap 比较时，直接按 `row_id` 对齐 |
| `outputs/predictions/test_<model>.csv` | 11,668 行，`row_id,prediction` | 各模型的最终测试预测，交给 wyt 统一评估 |
| `outputs/metrics/val_metrics.csv` | 每个模型的验证 MAE、MSE 及 95% CI，以及调参依据 | 主结果表（验证） |
| `outputs/metrics/test_metrics.csv` | 每个模型的测试 MAE、MSE 及 95% CI，以及调参依据 | 主结果表（测试） |
| `outputs/metrics/test_metrics_by_period.csv` | 每个模型 × {整体、第一年、之后}：行数、用户数、MAE、MSE | 整体结果与时间漂移对照 |
| `outputs/tuning/params.json` | 选定的 `user_mean_k`、`user_median_k`、Ridge 参数和偏移量 | 测试部分直接用这组参数重训，保证测试集上不再调参 |
| `outputs/tuning/val_user_k_grid.csv` | k ∈ {0, 1, 2, 3, 5, 10, 20, 50} 时两个用户模型的 MAE、MSE，以及预测低于 5 的比例 | 画 k 的调参曲线，说明正则化的作用 |
| `outputs/tuning/val_ridge_grid.csv` | 30 组 Ridge 超参数的验证 MSE、MAE、词表大小、预测标准差 | 比较两种食材表示和正则强度 |
| `outputs/tuning/val_ridge_shift_curve.csv` | 偏移量 c ∈ [−1, 1] 对应的验证 MAE | 画图说明 MAE 校准为什么退化成全部预测 5 |
| `outputs/tuning/ridge_coefficients.csv` | 全部食材词的 Ridge 系数，已排序 | 可解释性分析，即上面的系数表 |

所有输出都可以通过重新运行一遍 notebook 完整生成。

