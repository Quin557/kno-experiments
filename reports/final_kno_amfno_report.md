# KNO on AM-FNO Datasets: Final Experiment Report

## 1. Executive Summary

本项目以 KNO（Koopman Neural Operator）为主体模型，在 AM-FNO 使用或复现过的数据上进行实验，并与 AM-FNO 论文结果及本地 AM-FNO 复现结果进行对比。实验覆盖：

- NS-2D v1e-4：`ns_V1e-4_N10000_T30.mat`
- NS-2D v1e-3：`ns_V1e-3_N5000_T50.mat`
- CFD-1D：`1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5`
- CFD-2D：`2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`

最终结论：

1. KNO 在 NS-2D v1e-4 上稳定收敛，`test_full_rel_l2 = 6.2351e-02`，优于 AM-FNO 论文表中 NS-2D 数值 `8.51e-02`，但弱于本地 AM-FNO 复现结果 `2.4848e-02`。由于论文与本地复现的数据切分/版本可能不同，该结论应谨慎表述。
2. KNO 在 NS-2D v1e-3 上得到很低误差，`test_full_rel_l2 = 3.5535e-03`，可作为 KNO 在另一 NS 文件上的补充结果；但它不应直接与 v1e-4 的 AM-FNO 本地复现横向比较。
3. CFD-1D 初始未归一化版本训练失败，最终误差接近 `1.0`。加入训练集逐变量标准化后，KNO 稳定收敛到 `test_full_rel_l2 = 3.5024e-02`，但仍弱于 AM-FNO 本地复现的 `1.5164e-02`。
4. CFD-2D 标准 KNO 配置稳定收敛到 `test_full_rel_l2 = 4.4433e-02`，但与 AM-FNO 本地复现 `2.7686e-03` 差距较大。
5. CFD-2D 的两组 `o=64` 调参没有达到继续训练门槛。扩大 operator/latent size 能略微改善 100 epoch 内的最佳值，但引入较强震荡，不能有效缩小与 AM-FNO 的差距。

因此，本项目可给出的主要实验结论是：KNO 可以在 AM-FNO 的 NS 与 CFD 数据上完成稳定训练与多步 rollout；NS-2D 表现较好，CFD-1D 在加入逐变量归一化后可用，但 CFD-2D 与 AM-FNO 差距明显。对于 CFD 类多变量数据，KNO 原始 compact 结构需要更强的数据适配、变量尺度处理或模型结构改造，才能接近 AM-FNO。

## 2. Method

### 2.1 Model

本实验没有直接依赖服务器上的 `import koopmanlab`，因为服务器环境中出现过 `numpy.lib.arraypad`、`torchvision`、`h5py/numpy` 兼容问题。最终采用 `experiments/train_kno_amfno.py` 中的 standalone compact KNO 实现，结构与 KNO 原始代码保持一致：

```text
input
  -> encoder
  -> tanh observation space
  -> repeated Fourier-domain Koopman operator
  -> pointwise convolution high-frequency complement
  -> decoder prediction head
```

KNO 的核心思想是：不直接假设原始物理状态的时间演化是线性的，而是通过 encoder 学习观测空间，在该空间中使用 Koopman operator 进行近似线性推进。代码中每次 forward 内部重复 `decompose=8` 次 Koopman 更新，这对应 KNO 论文中的时间分解/时间复合思想。

### 2.2 NS-2D Input

NS-2D 使用单变量场 `u`，输入和输出均接近 KNO 原始 compact 设置：

| Item | Value |
|---|---|
| input | `[B, 64, 64, 10]` |
| target | `[B, 64, 64, 10]` |
| model | `KNO2dFlex(in_dim=10, out_dim=1)` |
| rollout | 自回归预测 10 步 |
| loss | `5.0 * prediction_mse + 0.5 * reconstruction_mse` |

### 2.3 CFD Input

CFD 是多变量序列，因此使用多变量适配：

| Dataset | Variables | Input | One-step Output |
|---|---|---|---|
| CFD-1D | density, pressure, Vx | `[B, X, 10, 3] -> [B, X, 30]` | `[B, X, 3]` |
| CFD-2D | density, pressure, Vx, Vy | `[B, X, Y, 10, 4] -> [B, X, Y, 40]` | `[B, X, Y, 4]` |

CFD 数据中的 density、pressure、velocity 数值尺度差异较大。首个 CFD-1D 未归一化 full run 失败后，训练脚本加入训练集逐变量标准化：

```text
x_norm = (x - mean_train_variable) / std_train_variable
```

训练损失在标准化空间计算，`metrics.csv` 中的 relative L2 则反标准化回原始物理量尺度后计算。因此最终结果仍可与 AM-FNO 的 relative L2 指标对比。

### 2.4 Metrics

报告主要使用：

- `test_step_rel_l2`：逐步预测误差的平均 relative L2。
- `test_full_rel_l2`：完整 rollout 窗口上的 relative L2。

主结果以最佳 `test_full_rel_l2` 为准，同时报告最终 epoch 数值用于判断稳定性。

## 3. Experimental Setup

### 3.1 Environment

所有正式实验均在远程服务器单卡运行：

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 |
| CUDA visible device | `CUDA_VISIBLE_DEVICES=1` |
| Python | 3.10.20 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| Run mode | `nohup python -u ...` |

### 3.2 Shared Hyperparameters

主实验默认配置：

| Hyperparameter | NS-2D | CFD-1D | CFD-2D |
|---|---:|---:|---:|
| epochs | 500 | 500 | 500 |
| `o` | 32 | 32 | 32 |
| modes | 16 | 16 | 16 |
| decompose | 8 | 8 | 8 |
| lr | 1e-3 | 1e-3 | 1e-3 |
| scheduler | StepLR, step 100, gamma 0.5 | same | same |
| weight decay | 1e-4 | 1e-4 | 1e-4 |
| batch size | 10 | 32 | 8 |
| grad clipping | none | 1.0 | 1.0 |

CFD 数据预处理：

| Dataset | reduced_resolution | reduced_resolution_t | reduced_batch | test_ratio |
|---|---:|---:|---:|---:|
| CFD-1D | 8 | 5 | 5 | 0.1 |
| CFD-2D | 2 | 1 | 5 | 0.1 |

## 4. Results

### 4.1 Completed KNO Runs

| Benchmark | Run | Best Epoch | Best Step Rel L2 | Best Full Rel L2 | Final Full Rel L2 | Time |
|---|---|---:|---:|---:|---:|---:|
| NS-2D v1e-4 | `kno_ns2d_v1e4_o32_m16_r8_ep500_seed42` | 486 | `5.5660e-02` | `6.2351e-02` | `6.2378e-02` | 2.0109 h |
| NS-2D v1e-3 | `kno_ns2d_v1e3_o32_m16_r8_ep500_seed42` | 461 | `3.5023e-03` | `3.5535e-03` | `4.0387e-03` | 2.3941 h |
| CFD-1D, no norm | `kno_cfd1d_o32_m16_r8_ep500_seed42` | 23 | `4.2286e-01` | `4.2512e-01` | `9.9900e-01` | 0.7140 h |
| CFD-1D, norm | `kno_cfd1d_norm_o32_m16_r8_ep500_seed42` | 482 | `3.4501e-02` | `3.5024e-02` | `3.5331e-02` | 1.1737 h |
| CFD-2D, norm | `kno_cfd2d_norm_o32_m16_r8_ep500_seed42` | 499 | `4.2271e-02` | `4.4433e-02` | `4.4433e-02` | 4.1589 h |
| CFD-2D, o64 | `kno_cfd2d_norm_o64_m16_r8_ep100_seed42` | 87 | `5.9974e-02` | `6.3840e-02` | `1.7205e-01` | 0.9549 h |
| CFD-2D, o64 low-lr | `kno_cfd2d_norm_o64_m16_lr5e4_gn05_ep100_seed42` | 92 | `5.9616e-02` | `6.2946e-02` | `9.0109e-02` | 0.9557 h |

### 4.2 Comparison With AM-FNO

| Benchmark | AM-FNO Paper | AM-FNO Local Repro | KNO Result | Assessment |
|---|---:|---:|---:|---|
| NS-2D v1e-4 | `8.51e-02` full/main | `2.4848e-02` full | `6.2351e-02` full | Better than paper table value, worse than local repro. Data split/version caveat applies. |
| CFD-1D | `1.47e-02` step | `1.5164e-02` full / `1.4850e-02` step | `3.5024e-02` full / `3.4501e-02` step | Stable after normalization, but about 2.3x worse than local AM-FNO. |
| CFD-2D | `2.16e-03` step | `2.7686e-03` full / `2.7442e-03` step | `4.4433e-02` full / `4.2271e-02` step | Stable, but about 16x worse than local AM-FNO full metric. |

NS-2D v1e-3 不列入 AM-FNO 主对比表，因为当前本地 AM-FNO 复现摘要使用的是 v1e-4 文件。该实验保留为 KNO 补充结果。

## 5. Analysis

### 5.1 NS-2D

NS-2D 是最接近 KNO 原始实验接口的数据类型：单变量输入窗口，模型输出下一步单变量状态，再自回归滚动。因此 KNO 在 NS-2D 上的结果最健康。

v1e-4 实验最终与最佳值几乎一致：

```text
best full: 6.2351e-02 at epoch 486
final full: 6.2378e-02 at epoch 499
```

这说明训练后期没有明显过拟合或退化。v1e-3 结果更低：

```text
best full: 3.5535e-03 at epoch 461
final full: 4.0387e-03 at epoch 499
```

但 v1e-3 与 v1e-4 的数据文件、粘性系数和样本设置不同，不能简单用数值大小判断模型优劣。

### 5.2 CFD-1D

CFD-1D 是本项目中最能说明数据适配重要性的实验。未归一化版本出现明显失败：

```text
best full: 4.2512e-01 at epoch 23
final full: 9.9900e-01 at epoch 499
```

加入训练集逐变量标准化后，结果变为：

```text
best full: 3.5024e-02 at epoch 482
final full: 3.5331e-02 at epoch 499
```

这说明原始失败主要不是 KNO 完全不能学习 CFD-1D，而是多物理变量的尺度差异影响了训练。归一化后模型能够稳定 rollout，但仍未达到 AM-FNO 水平。

### 5.3 CFD-2D

CFD-2D 的主配置 `o=32` 是稳定的，并且最佳点出现在最后一轮：

```text
epoch 0 full:   3.0062e-01
epoch 100 full: 6.5971e-02
epoch 300 full: 5.1015e-02
epoch 499 full: 4.4433e-02
```

这说明模型仍在慢慢学习，但收敛速度和最终精度都明显落后于 AM-FNO。

两组 `o=64` 调参结果如下：

| Run | Best Full Rel L2 | Final Full Rel L2 | Decision |
|---|---:|---:|---|
| `o=64, lr=1e-3, grad_norm=1.0` | `6.3840e-02` | `1.7205e-01` | 不继续 |
| `o=64, lr=5e-4, grad_norm=0.5` | `6.2946e-02` | `9.0109e-02` | 不继续 |

低学习率和更强裁剪让最终值从 `1.7205e-01` 改善到 `9.0109e-02`，但最佳值只从 `6.3840e-02` 改善到 `6.2946e-02`，仍未达到预设继续训练门槛 `5.5e-02`。因此不建议继续 `o=64` 500 epoch。

### 5.4 AM-FNO 与 KNO 的关键差异及误差来源

AM-FNO 和 KNO 都在 Fourier 空间中学习算子，但二者的建模目标和参数使用方式不同。这也是本次实验中 NS-2D 可比较、CFD 尤其是 CFD-2D 差距明显的主要模型层面原因。

| Aspect | AM-FNO local reproduction | KNO implementation in this project | Effect in this experiment |
|---|---|---|---|
| Fourier kernel | 使用 FNO 主干，4 个 spectral block；MLP/KAN 根据频率坐标生成复数 Fourier kernel。 | encoder 将输入映射到观测空间；低频 Fourier modes 上使用可学习 Koopman matrix。 | AM-FNO 的 kernel 更直接面向空间频率响应；KNO 的 kernel 更强调观测空间时间推进。 |
| Time modeling | 对输入窗口到下一帧/输出变量做 supervised mapping，再 rollout。 | 在观测空间中重复 `decompose=8` 次 Koopman update，强调同一线性推进算子的时间复合。 | KNO 对 NS-2D 这类单变量演化较自然；CFD 多变量耦合需要额外适配。 |
| Spatial coordinates | AM-FNO 将 grid 坐标拼到输入通道，1D 加 `x`，2D 加 `(x,y)`。 | 当前 KNO 训练脚本读取 grid，但模型 forward 未显式使用 grid。 | CFD-2D 中空间位置和边界/周期结构信息对局部模式有帮助，KNO 少了这一显式条件。 |
| Multi-variable handling | 输入维度直接包含历史窗口、变量和坐标，并通过 FNO block 混合。 | CFD 中将 `initial_step * variables` 压平为通道，再用线性 encoder 映射到 `o` 维。 | KNO 的变量耦合主要依赖单个线性 encoder，缺少变量专用 encoder 或 cross-variable block。 |
| High-frequency/local detail | 每个 spectral block 后有 pointwise MLP，且 Fourier kernel 覆盖完整频率网格。 | 仅保留 `modes=16` 的低频 Koopman matrix，另有 1x1 conv high-frequency complement。 | CFD-2D 的局部高频和变量耦合更复杂，KNO 的高频补偿能力不足。 |
| Loss design | 主要优化预测 relative L2/rollout 任务。 | 使用 `5 * prediction_mse + 0.5 * reconstruction_mse`，还要约束观测空间可重构。 | reconstruction loss 有助于 KNO 表示稳定，但也会占用容量；CFD 精度任务上不一定最优。 |

参数量与训练成本对比如下。AM-FNO 数字来自本地复现 summary 日志，KNO 数字来自本项目 `outputs/*/env.txt` 与 `metrics.csv`。

| Benchmark | AM-FNO Params | KNO Params | Param Ratio | AM-FNO Avg Sec/Epoch | KNO Avg Sec/Epoch | Cost Observation |
|---|---:|---:|---:|---:|---:|---|
| NS-2D v1e-4 | 1,136,929 | 526,059 | AM-FNO 2.16x KNO | 11.06 | 14.48 | KNO 参数更少但略慢，主要由于自回归 rollout 和复数 Koopman 操作。 |
| CFD-1D | 588,419 | 35,905 | AM-FNO 16.39x KNO | 9.34 | 8.45 | KNO 极小模型已可稳定训练，但容量明显不足，精度约差 2.3x。 |
| CFD-2D | 2,286,180 | 528,108 | AM-FNO 4.33x KNO | 14.29 | 29.94 | KNO 参数更少但约 2.10x 慢；CFD-2D 中计算瓶颈来自 2D FFT、复数矩阵乘和 11-step rollout。 |

这组对比说明，CFD-2D 差距不是简单的“参数越多越好”。AM-FNO 的参数更多，但这些参数主要服务于多层频域 kernel 生成和局部 pointwise mixing，直接增强空间频率响应；KNO 的参数较少，集中在观测空间 encoder/decoder 与 Koopman matrix，适合学习可复合的时间推进，但对 CFD-2D 的多变量局部结构表达不足。

从模型设计看，本次 CFD 差距主要来自四点：

1. **变量耦合不足。** CFD-2D 的 density、pressure、Vx、Vy 之间存在强耦合；当前 KNO 只是把 10 帧历史和 4 个变量压平到 40 个通道，再用一个线性 encoder 混合，缺少 AM-FNO 那种多层 spectral block 的反复交互。
2. **空间条件不足。** AM-FNO 显式拼接 grid 坐标；当前 KNO 没有把 grid 输入模型。对周期 CFD 数据这不是致命问题，但会削弱模型对空间结构和局部变化的条件化能力。
3. **低频 Koopman 偏置。** KNO 的 Koopman matrix 只作用在低频 modes 上，高频部分主要靠 1x1 conv 补偿；CFD-2D 的小尺度涡旋/激波样变化更依赖局部与高频表达。
4. **容量分配不同。** CFD-2D 主 KNO 只有约 52.8 万参数，而 AM-FNO 约 228.6 万参数。扩大 KNO 到 `o=64` 后参数量接近 AM-FNO（约 210.7 万），但早期误差只小幅改善且震荡更强，说明问题不只是容量，而是容量放在了不够适合 CFD-2D 的 Koopman latent matrix 上。

因此，若后续希望缩小 CFD-2D 差距，优先方向不是继续盲目增大 `o`，而是改模型结构：加入 grid conditioning、变量专用 encoder、跨变量 mixing、强局部卷积分支，或设计 rollout-aware loss。当前结果更适合解释为：KNO 的时间复合优势在 NS-2D 上比较自然，但原始 compact KNO 迁移到 AM-FNO 的多变量 CFD 任务时，需要额外结构改造。

## 6. Discussion

### 6.1 What Worked

- Standalone KNO 训练脚本避免了服务器上 KoopmanLab 包安装问题。
- NS-2D 与 KNO 原始输入假设接近，训练稳定。
- CFD 逐变量标准化是必要且有效的，尤其对 CFD-1D 改善显著。
- 使用 `nohup python -u` 后，日志与 `metrics.csv` 记录完整，适合 SSH 断开后的远程实验。

### 6.2 What Did Not Work

- CFD-1D 未归一化版本不可用。
- CFD-2D 虽然稳定，但与 AM-FNO 差距很大。
- 简单扩大 `o` 到 64 并不能解决 CFD-2D 问题，反而引入更明显的测试震荡。

### 6.3 Likely Reasons For CFD Gap

可能原因包括：

1. KNO 原始 compact 结构更适合单变量时序场，CFD 多变量状态需要更细的变量耦合建模。
2. AM-FNO 针对多尺度频域特征和 amortized 参数生成设计，更适合 CFD-2D 中复杂空间变化。
3. 当前 KNO 仅使用低频 Koopman modes 加 pointwise high-frequency complement，可能不足以捕捉 CFD-2D 中的局部高频结构。
4. CFD-2D 的 full rollout 对误差累积更敏感，当前自回归训练没有使用 scheduled sampling、teacher forcing 混合策略或 rollout-aware 稳定化。

## 7. Final Recommendation

本阶段实验可以收束，不建议继续大范围调参。最终报告主结果建议使用：

| Benchmark | Final KNO Run |
|---|---|
| NS-2D v1e-4 | `kno_ns2d_v1e4_o32_m16_r8_ep500_seed42` |
| NS-2D v1e-3 | `kno_ns2d_v1e3_o32_m16_r8_ep500_seed42` |
| CFD-1D | `kno_cfd1d_norm_o32_m16_r8_ep500_seed42` |
| CFD-2D | `kno_cfd2d_norm_o32_m16_r8_ep500_seed42` |

`kno_cfd1d_o32_m16_r8_ep500_seed42` 应作为未归一化失败对照。两组 CFD-2D `o=64` run 应作为调参负例，而不是主结果。

如果后续继续研究，优先方向不是继续扩大 `o`，而是：

1. 为 CFD 多变量输入加入变量专用 encoder 或 cross-variable mixing block。
2. 在 KNO 频域分支之外加入更强的局部卷积分支。
3. 使用 rollout-aware loss，对长时间预测误差直接加权。
4. 对 CFD-2D 尝试变量分组归一化、物理量加权 loss 或按变量分别报告误差。
5. 在多卡空闲时做小规模系统搜索，但以结构改造为主，而非只调 `o/modes`。

## 8. Reproducibility Notes

关键文件：

- Training script: `experiments/train_kno_amfno.py`
- Server checklist: `docs/server_run_checklist.md`
- Baseline table: `results/amfno_reference_baselines.csv`
- Run summary: `results/kno_run_summary.csv`
- Current comparison summary: `results/kno_vs_amfno_summary.md`

数据文件较大，没有提交到 Git。`.gitignore` 已忽略：

```text
ref/
*.h5
*.hdf5
*.mat
*.npy
*.npz
*.pt
*.pth
*.ckpt
```

远程服务器数据路径由私有文件 `configs/data_paths.env` 指定，该文件不会进入 Git。

## 9. Appendix: Main Run Commands

### NS-2D v1e-4

```bash
nohup python -u experiments/train_kno_amfno.py \
  --benchmark ns2d \
  --ns-file "$DATA_ROOT/$NS2D_V1E4" \
  --run-name kno_ns2d_v1e4_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 10 --seed 42 \
  --o 32 --modes 16 --decompose 8 \
  --t-in 10 --t-out 10 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_ns2d_v1e4_ep500_seed42.log 2>&1 &
```

### NS-2D v1e-3

```bash
nohup python -u experiments/train_kno_amfno.py \
  --benchmark ns2d \
  --ns-file "$DATA_ROOT/$NS2D_V1E3" \
  --run-name kno_ns2d_v1e3_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 10 --seed 42 \
  --o 32 --modes 16 --decompose 8 \
  --t-in 10 --t-out 10 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_ns2d_v1e3_ep500_seed42.log 2>&1 &
```

### CFD-1D

```bash
nohup python -u experiments/train_kno_amfno.py \
  --benchmark cfd1d \
  --cfd-file "$DATA_ROOT/$CFD1D_FILE" \
  --run-name kno_cfd1d_norm_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 32 --seed 42 \
  --o 32 --modes 16 --decompose 8 \
  --initial-step 10 --t-train 21 \
  --reduced-resolution 8 --reduced-resolution-t 5 --reduced-batch 5 \
  --cfd-normalize --max-grad-norm 1.0 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_cfd1d_norm_ep500_seed42.log 2>&1 &
```

### CFD-2D

```bash
nohup python -u experiments/train_kno_amfno.py \
  --benchmark cfd2d \
  --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name kno_cfd2d_norm_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 8 --seed 42 \
  --o 32 --modes 16 --decompose 8 \
  --initial-step 10 --t-train 21 \
  --reduced-resolution 2 --reduced-resolution-t 1 --reduced-batch 5 \
  --cfd-normalize --max-grad-norm 1.0 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_cfd2d_norm_ep500_seed42.log 2>&1 &
```
