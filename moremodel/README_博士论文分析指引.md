# 博士论文：高维临床数据分析与预测模型

本目录为你的 **`test_data.xlsx`**（160 例 × 11016 指标）准备了完整流程，无需建模基础也可按步骤运行。

---

## 第一步：安装依赖（只需做一次）

在终端进入本目录后执行：

```bash
cd /Users/samuels/riskpredict/moremodel
pip install -r requirements.txt
```

---

## 第二步：数据预处理（请先完成）

```bash
python run_thesis_pipeline.py
```

或：

```bash
python preprocess_highdim_clinical.py
```

### 你会得到什么？

文件夹 **`preprocessed/`**：

| 文件 | 用途 |
|------|------|
| **X_preprocessed.csv** | 30 个入选指标，已标准化（写论文「预处理结果」、部分探索性分析） |
| **y.csv** | 结局：0=未发病，1=发病 |
| **预处理说明.md** | 中文说明，可直接改写入论文「资料与方法」 |
| univariate_scores.csv | 每个原始指标与结局的相关性（备查） |
| feature_list.txt | 30 个入选指标名称列表 |

### 预处理在做什么？（一句话）

从 1 万多个指标里，按统计学规范挑出 **30 个** 与「是否 QFR≤0.8」最相关、且彼此不太重复的指标，并做标准化。

---

## 第三步：构建预测模型（论文「结果」用）

预处理完成后运行：

```bash
python run_thesis_pipeline.py --step model
```

或一步完成预处理 + 建模：

```bash
python run_thesis_pipeline.py --step all
```

### 建模输出（文件夹 `thesis_results/`）

| 文件 | 用途 |
|------|------|
| **论文_统计学方法_草稿.md** | 统计学方法段落草稿 |
| **论文_结果_草稿.md** | 结果段落草稿（含 AUC、各折表格） |
| **roc_curve_cv.png** | ROC 曲线图（可直接插入论文） |
| thesis_model_report.json | 交叉验证 AUC、各折详情 |
| cv_oof_predictions.csv | 每个样本的预测概率（可画 ROC） |
| final_model_lasso_coefficients.csv | LASSO 系数（哪些指标方向性重要） |

**重要**：正式报告模型好坏时，以 **`cv_auc`（交叉验证 AUC）** 为准，不要用「在全数据上训练再评估」的数值。

---

## 论文里怎么写？（结构建议）

1. **资料与方法**  
   - 样本量、结局定义  
   - 复制/改写 `preprocessed/预处理说明.md` 中的预处理流程  
   - 复制/改写 `thesis_results/论文_统计学方法_草稿.md` 中的模型与验证部分  

2. **结果**  
   - 报告交叉验证 AUC（见 `thesis_model_report.json`）  
   - 可附入选特征表（`feature_list.txt`）  

3. **讨论**  
   - 说明 p>>n、需外部验证、过拟合风险  

---

## 常见问题

**Q：要不要自己算 1 万个指标的两两相关？**  
A：不需要。脚本只在「与结局最相关的前 2000 个」子集上处理共线性。

**Q：为什么只留 30 个指标？**  
A：阳性约 71 例，变量过多易过拟合；30 是在筛选信息量与稳定性的折中，且配合 LASSO 正则化。

**Q：预处理数据和建模用的数据一样吗？**  
A：`preprocessed/` 是在全数据上做一次探索性筛选；**建模脚本会在每一折交叉验证的训练折里重新筛选**，更符合论文规范。

---

## 联系与修改参数

如需调整保留特征数（例如改为 20）：

```bash
python run_thesis_pipeline.py --top-k 20 --step all
```

如有导师指定的缺失率阈值、折数等，可同样加参数运行；需要说明可再沟通。

---

## 第四步：继续优化（当前推荐模型）

```bash
python final_optimize.py
python run_external_validation.py --test verify.xlsx --model optimized_model/final_model.joblib --output-dir test_results_final
```

当前最优约 **verify AUC ≈ 0.72**（SVM-RBF + 15 个稳定特征），详见 `optimized_model/最终优化说明.md`。

---

## 训练集 AUC 优化（早期阶段）

在确认独立测试集之前，可先拉高**训练集样本内 AUC**：

```bash
python optimize_train_model.py
```

输出目录 **`trained_model/`**：

| 文件 | 说明 |
|------|------|
| `best_model.joblib` | 保存的模型（测试集预测用） |
| `feature_list.txt` | 固定特征名（约 80 个） |
| `训练集优化说明.md` | 优化前后 AUC 对比 |

独立测试集到达后：

```bash
python predict_external_test.py --test /path/to/你的测试集.xlsx
```
