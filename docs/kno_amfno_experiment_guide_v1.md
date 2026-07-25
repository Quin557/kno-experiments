# KNO on AM-FNO Data: Experiment Guide v1

## 0. Goal

本实验的目标是：以 KNO 为主体模型，使用 AM-FNO 已使用/已复现的数据集，跑出 `CFD-1D`、`CFD-2D`、`NS-2D` 三组主实验，并和 AM-FNO 论文结果以及已有 AM-FNO 复现结果进行比较。

本阶段先完成四件事：

1. 读清楚 KNO 论文和 KoopmanLab 代码。
2. 明确 AM-FNO 三个目标数据集的输入/输出张量。
3. 给出服务器实验准备、运行、记录、上传规范。
4. 为下一步写 KNO 数据适配和训练脚本确定接口。

## 1. Source Materials

本地参考材料：

- `ref/KNO.pdf`: Koopman Neural Operator 原论文。
- `ref/Xiao ... - Amortized Fourier Neural Operators.pdf`: AM-FNO 原论文。
- `ref/AM-FNO-repro-report/AM-FNO_MLP_report_v4.docx`: AM-FNO 复现报告格式和复现结果参考。
- `external/KoopmanLab`: 已本地克隆的 KNO 官方开源仓库。
- `external/am_fno_repro`: 已本地克隆的 AM-FNO 复现仓库。

远程仓库：

- KNO: `https://github.com/Koopman-Laboratory/KoopmanLab`
- AM-FNO reproduction: `https://github.com/Quin557/am_fno_repro`
- 当前实验仓库: `git@github.com:Quin557/kno-experiments.git`

## 2. KNO Paper Summary

KNO 的核心观点不是简单地“换一个 Fourier 层”，而是把 PDE 解的时间演化看成一个动态系统，再通过 Koopman operator 在观测空间里把非线性演化转成线性推进问题。

论文中的关键概念：

- PDE 解序列可被看作状态 `gamma_t` 随时间演化的动态系统。
- 原始状态空间中的演化通常是非线性的，因此长时间预测会不断累积误差。
- Koopman operator 作用在观测函数 `g(gamma_t)` 上；在合适的观测空间里，`g(gamma_t)` 的演化可近似为线性的。
- KNO 用神经网络学习观测函数、低频 Fourier 空间中的 Koopman 线性推进、高频补偿和反观测映射。

论文给出的 KNO 模型流程可以拆成六部分：

1. Observation: 用 encoder 把输入状态 `phi_t` 映射到观测空间 `g(gamma_t)`。
2. Fourier transform: 对观测特征做 FFT，并截断/保留低频 modes。
3. Koopman operator: 在 Fourier 空间中学习一个复数线性矩阵，对低频观测特征做时间推进。
4. Inverse Fourier transform: 把推进后的低频观测特征转回物理空间。
5. High-frequency complement: 用卷积网络补偿被低频 Koopman 分支过滤掉的高频变化。
6. Inverse observation: 用 decoder 把观测空间结果映射回目标物理状态。

KNO 的主要实验设计：

- Mesh-independent: 在不同空间分辨率上训练/测试，验证 neural operator 的跨网格能力。
- Long-term prediction: 在 NS、Rayleigh-Benard、shallow-water、真实水汽/洋流等数据上做多步预测。
- Zero-shot resolution: 低分辨率训练，高分辨率测试。
- Zero-shot prediction interval: 在未监督时间间隔上做插值/外推。
- Ablation: 去掉 reconstruction loss，验证 Koopman-like 线性层是否真的有贡献。

对本项目最重要的结论：

- KNO 的优势重点在长时间 rollout 和时间间隔泛化，而不只是单步误差。
- KNO 使用 `prediction loss + reconstruction loss`，其中 reconstruction loss 迫使 encoder/decoder 学到可逆的观测表示，否则中间线性层未必像 Koopman operator。
- KNO 默认实验并没有直接覆盖 AM-FNO 的 CFD-1D/CFD-2D 数据，因此需要写数据适配和多变量输出处理。

## 3. AM-FNO Baselines To Compare

AM-FNO 论文 Table 2 中与本项目相关的数值：

| Benchmark | AM-FNO(MLP) paper | AM-FNO(KAN) paper | Strong baseline in paper |
|---|---:|---:|---:|
| NS-2D | `8.51e-2` | `1.08e-1` | U-FNO `1.22e-1` |
| CFD-1D | `1.47e-2` | `1.83e-2` | U-FNO `2.44e-2` |
| CFD-2D | `2.16e-3` | `2.70e-3` | U-FNO `4.52e-3` |

AM-FNO 复现报告中已有结果：

| Benchmark | Repro metric | Repro result | Note |
|---|---|---:|---|
| NS-2D | `test_step_rel_l2 / test_full_rel_l2` | `1.936595e-02 / 2.484761e-02` | 使用 `ns_V1e-4_N10000_T30.mat`，与论文切分/版本可能不同，不直接判断优劣。 |
| CFD-1D | `test_step_rel_l2 / test_full_rel_l2` | `1.484995e-02 / 1.516407e-02` | step 指标基本对齐论文 `1.47e-2`。 |
| CFD-2D | `test_step_rel_l2 / test_full_rel_l2` | `2.744214e-03 / 2.768594e-03` | 同量级但未达到论文 `2.16e-3`；batch=32 明显优于早期 batch=8。 |

建议本项目最终报告同时放三列：

- AM-FNO paper。
- AM-FNO local reproduction。
- KNO on same local data。

这样可以避免“论文数据口径”和“本地数据口径”混在一起。

## 4. KoopmanLab Code Reading

KoopmanLab compact KNO 的核心代码非常集中：

- `koopmanlab/models/kno.py`: 模型结构。
- `koopmanlab/model.py`: 封装 compile、optimizer、train/test。
- `koopmanlab/data.py`: Burgers、shallow water、Navier-Stokes 数据接口。
- `demo_ns.py`: NS-2D 运行示例。

### 4.1 Core model files

`kno.py` 中的核心类：

- `encoder_mlp`, `decoder_mlp`: 对最后一维 `t_len` 做线性升维/降维。
- `encoder_conv1d`, `decoder_conv1d`: 1D pointwise conv autoencoder。
- `encoder_conv2d`, `decoder_conv2d`: 2D pointwise conv autoencoder。
- `Koopman_Operator1D`: 对 `[B, op_size, X]` 做 `rfft`，低频 modes 上用复数 Koopman matrix 线性推进，再 `irfft`。
- `KNO1d`: encoder -> tanh -> repeated Koopman update -> high-frequency conv complement -> decoder。
- `Koopman_Operator2D`: 对 `[B, op_size, X, Y]` 做 `rfft2`，低频 modes 上做复数矩阵推进，再 `irfft2`。
- `KNO2d`: 2D 版本的 KNO 主体。

### 4.2 Why "it learns matrices" but can compose over time

这个问题的关键在于：KNO 学的矩阵不是直接在原始物理状态 `u(x,t)` 上做一步线性回归，而是在神经网络构造出的观测空间 `g(u)` 上做线性推进。

直观理解：

- 原始空间：`u_t -> u_{t+1}` 通常高度非线性。
- 观测空间：通过 encoder 得到 `z_t = g(u_t)`。
- Koopman 空间假设：存在某种观测 `g`，使得 `z_{t+1} ≈ K z_t`。
- 多步预测：如果一步是 `K`，那么两步是 `K^2`，十步是 `K^10`。这就是时间复合的来源。

代码里具体体现为：

```text
for i in range(decompose):
    x1 = koopman_layer(x)
    x = x + x1
```

这里 `koopman_layer` 是 Fourier 空间中的复数矩阵乘法，`decompose/r` 相当于在一次 forward 内部做多次较短 Koopman 推进。论文解释说，如果数据采样时间间隔比较粗，把一个时间间隔拆成 `r` 次内部推进，可以让每次 Koopman operator 面对的时间变化更小，从而更容易学习。

因此它“能时间复合”的核心不是矩阵本身神奇，而是：

1. encoder 把非线性状态搬到更适合线性演化的观测空间；
2. Koopman matrix 在 Fourier 低频空间做线性推进；
3. `decompose/r` 让同一个推进模块可以重复作用；
4. rollout 时把预测的新帧滑入输入窗口，继续预测下一帧；
5. reconstruction loss 约束观测空间不要退化成只服务单步预测的黑箱表示。

### 4.3 KNO input and output shapes

KoopmanLab README 和代码默认 compact KNO 的张量形状：

- KNO1d input: `[B, X, t_in]`
- KNO1d output: `[B, X, t_in]`
- KNO2d input: `[B, X, Y, t_in]`
- KNO2d output: `[B, X, Y, t_in]`

注意：原始 compact KNO 的 decoder 会输出 `t_in` 个通道，然后训练循环通常只取 `im[..., -1:]` 作为下一步预测。因此对于 NS-2D 这类单变量序列，`t_in=10` 很自然：输入前 10 帧，模型输出一个长度为 10 的窗口，其中最后一帧作为下一帧。

但是 CFD-1D/CFD-2D 是多变量时间序列：

- CFD-1D: 每个时间步有 3 个变量 `[density, pressure, Vx]`。
- CFD-2D: 每个时间步有 4 个变量 `[density, pressure, Vx, Vy]`。

AM-FNO 的处理方式是把 `initial_step * variables` 压成输入通道：

- CFD-1D input: `[B, X, 10, 3] -> [B, X, 30]`
- CFD-2D input: `[B, X, Y, 10, 4] -> [B, X, Y, 40]`

如果直接套 KNO compact API，会遇到一个问题：KNO 输出最后一维长度等于 `t_in`，而 CFD 需要下一步输出变量数 `3` 或 `4`。因此第一版适配有两个可选方案：

| Plan | Design | Pros | Cons |
|---|---|---|---|
| A | 把 KNO 的 `t_in` 视为压平通道，输出同样通道，再取最后 `C_vars` 个通道当下一帧 | 改动小，能最快跑通 | 语义不够干净，decoder 输出历史窗口通道而不是变量通道 |
| B | 新写 `KNO1dMultiVar/KNO2dMultiVar`，encoder 输入 `initial_step * C_vars`，decoder 输出 `C_vars` | 语义正确，更适合报告 | 需要改模型类和训练脚本 |

建议采用 Plan B。原因是本项目最终报告要讲清楚模型输入输出，Plan B 更容易自洽，也更便于和 AM-FNO 的 `input_dim/output_dim` 对齐。

## 5. Target Dataset Mapping

### 5.1 NS-2D

数据文件：

- `ns_V1e-3_N5000_T50.mat`
- `ns_V1e-4_N10000_T30.mat`

字段：

- `u`: 标量场序列，通常作为 NS vorticity/state。

建议设置：

| Item | Value |
|---|---|
| model | `KNO2d` |
| input shape | `[B, 64, 64, 10]` |
| target shape | `[B, 64, 64, 10]` |
| rollout | 10 steps for AM-FNO comparison; optionally 40 steps for KNO long-term strength |
| ntrain/ntest | v1e-3: `1000/200`; v1e-4: first `1000` train + last `200` test for AM-FNO-local comparison |
| metric | relative L2 step/full; optionally MSE for KNO-paper-style reporting |

### 5.2 CFD-1D

数据文件：

- `1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5`

字段：

- `density`
- `pressure`
- `Vx`
- `x-coordinate`

建议设置：

| Item | Value |
|---|---|
| model | `KNO1dMultiVar` |
| raw state | `[B, X, T, 3]` |
| input shape | `[B, X, 30]` from `10 * 3` |
| target single-step | `[B, X, 3]` |
| rollout | predict `t=10..20`, 11 steps when `t_train=21` |
| preprocessing | match AM-FNO: `reduced_resolution=8`, `reduced_resolution_t=5`, `reduced_batch=5`, `test_ratio=0.1` |
| metric | relative L2 step/full |

### 5.3 CFD-2D

数据文件：

- `2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`

字段：

- `density`
- `pressure`
- `Vx`
- `Vy`
- `x-coordinate`
- `y-coordinate`

建议设置：

| Item | Value |
|---|---|
| model | `KNO2dMultiVar` |
| raw state | `[B, X, Y, T, 4]` |
| input shape | `[B, 64, 64, 40]` from `10 * 4` |
| target single-step | `[B, 64, 64, 4]` |
| rollout | predict `t=10..20`, 11 steps when `t_train=21` |
| preprocessing | match AM-FNO: `reduced_resolution=2`, `reduced_resolution_t=1`, `reduced_batch=5`, `test_ratio=0.1` |
| metric | relative L2 step/full and optional frequency-region error |

## 6. Hyperparameter Policy

原则：KNO 为主体，尽量保持 AM-FNO 的数据切分、rollout、metric 口径一致；当 KNO 原论文设置和 AM-FNO 差异很大时，优先选择对 KNO 合理的设置，并在报告里解释原因。

KNO 原论文常用设置：

- optimizer: Adam
- learning rate: `0.001` or demo `0.005`
- scheduler: StepLR, every 100 epochs multiply `0.5`
- prediction loss/reconstruction loss: `5 * pred + 0.5 * recons`
- NS long-term demo: `o=32`, `m=16`, `r=8`, `t_in=10`
- batch size: NS demo `10`; 1D Burgers paper section uses `64`

第一版建议搜索空间：

| Dataset | o | modes | r/decompose | batch | epochs |
|---|---:|---:|---:|---:|---:|
| NS-2D v1e-3 | 32 | 16 | 8 | 10 | 100 smoke, 500 final |
| NS-2D v1e-4 | 32 | 16 | 8 | 10 | 100 smoke, 500 final |
| CFD-1D | 32 | 16 or 32 | 8 | 32 or 64 | 100 smoke, 500 final |
| CFD-2D | 32 | 16 | 8 | 8 or 16 | 100 smoke, 500 final |

如果 CFD-2D 显存压力小，可试 `o=64` 或 `m=24`；如果不稳定，先降低 batch，不要急着改 r。

## 7. Server Environment

服务器条件：8 x NVIDIA RTX A6000。

KNO compact model 原始代码没有 DDP/torchrun 训练入口。建议第一阶段采用单卡训练，但用 8 张卡并行跑不同实验：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_kno.sh ns2d_v1e3
CUDA_VISIBLE_DEVICES=1 bash scripts/run_kno.sh ns2d_v1e4
CUDA_VISIBLE_DEVICES=2 bash scripts/run_kno.sh cfd1d
CUDA_VISIBLE_DEVICES=3 bash scripts/run_kno.sh cfd2d
```

后续如果单个实验耗时过长，再考虑把训练循环改成 DDP。

推荐环境：

```bash
conda create -n kno-amfno python=3.10 -y
conda activate kno-amfno
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy h5py matplotlib tqdm pandas pyyaml
pip install -e external/KoopmanLab
```

如果服务器已有 PyTorch 环境，先验证：

```bash
python - <<'PY'
import torch, numpy, scipy, h5py
print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

## 8. Repository Layout

建议仓库结构：

```text
configs/
  data_paths.example.env
  data_paths.env              # private, ignored
docs/
  kno_amfno_experiment_guide_v1.md
experiments/
  train_kno_amfno.py          # next step
external/
  KoopmanLab/                 # local clone, ignored by this repo
  am_fno_repro/               # local clone, ignored by this repo
reports/
  final_report.md             # final report, commit
results/
  summary.csv                 # lightweight result summary, commit
scripts/
  run_kno.sh                  # next step
data/                         # large datasets, ignored
outputs/                      # raw run outputs, ignored
```

Data and checkpoints must not be committed. Only commit:

- code;
- config templates;
- lightweight CSV summaries;
- final plots after compression;
- final report.

## 9. Data Placement On Server

After cloning this repository on the server:

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
```

Recommended data tree:

```text
data/
  cfd/
    1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5
    2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5
  ns2d/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
```

Then set:

```bash
DATA_ROOT=/absolute/path/to/data
CFD1D_FILE=cfd/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5
CFD2D_FILE=cfd/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5
NS2D_V1E3=ns2d/ns_V1e-3_N5000_T50.mat
NS2D_V1E4=ns2d/ns_V1e-4_N10000_T30.mat
```

## 10. First Implementation Plan

下一步代码实现建议按这个顺序：

1. Copy AM-FNO `FNODatasetSingle` logic into this project, or import it as reference, so CFD preprocessing exactly matches reproduction.
2. Write `RelativeLpLoss`, so KNO and AM-FNO use the same relative L2 metric.
3. Implement `KNO1dMultiVar` and `KNO2dMultiVar` by modifying KoopmanLab compact KNO decoder output dimension.
4. Implement autoregressive train/eval loops matching AM-FNO:
   - NS: input `[B,H,W,10]`, output one frame, roll 10 steps.
   - CFD-1D: input `[B,X,10,3] -> [B,X,30]`, output `[B,X,3]`, roll 11 steps.
   - CFD-2D: input `[B,H,W,10,4] -> [B,H,W,40]`, output `[B,H,W,4]`, roll 11 steps.
5. Write `metrics.csv`, `args.json`, `env.txt`, and optional small prediction figure per run.
6. Run smoke tests with `epochs=1` and a tiny subset before full training.

## 11. Experiment Matrix

Minimum official runs:

| Run name | Dataset | File | GPU use | Main comparison |
|---|---|---|---|---|
| `kno_ns2d_v1e3_seed42` | NS-2D | `ns_V1e-3_N5000_T50.mat` | 1 GPU | KNO paper setting and AM-FNO-style metric |
| `kno_ns2d_v1e4_seed42` | NS-2D | `ns_V1e-4_N10000_T30.mat` | 1 GPU | local AM-FNO reproduction |
| `kno_cfd1d_seed42` | CFD-1D | given HDF5 | 1 GPU | AM-FNO paper/local |
| `kno_cfd2d_seed42` | CFD-2D | given HDF5 | 1 GPU | AM-FNO paper/local |

Optional robustness runs:

| Run name | Change |
|---|---|
| `kno_cfd2d_o64_seed42` | Increase operator size `o=64` |
| `kno_cfd2d_m24_seed42` | Increase Fourier modes |
| `kno_ns2d_v1e4_seed0/1/2` | Seed sensitivity |

## 12. Metrics And Records

Each run should create:

```text
outputs/<run_name>/
  args.json
  metrics.csv
  env.txt
  train.log
  checkpoint_last.pt          # ignored
  checkpoint_best.pt          # ignored
  pred_sample.pt              # ignored unless small
```

After full runs, copy only lightweight summaries to git:

```text
results/
  kno_main_results.csv
  kno_vs_amfno_summary.md
reports/
  experiment_report.md
```

Required metrics:

- `train_step_rel_l2`
- `train_full_rel_l2`
- `test_step_rel_l2`
- `test_full_rel_l2`
- epoch time
- total training time
- parameter count
- GPU model and memory

Optional metrics:

- per-step rollout error curve;
- CFD-2D low/mid/high frequency error, matching AM-FNO Table 3;
- MSE, for easier comparison with KNO original paper plots.

## 13. Reporting Structure

Final report can be reorganized as:

1. Motivation: why compare KNO and AM-FNO on these data.
2. KNO method: Koopman observation, Fourier-domain linear evolution, high-frequency complement.
3. Data and preprocessing: NS-2D, CFD-1D, CFD-2D.
4. Implementation adaptation: what changed from KoopmanLab.
5. Experimental setup: environment, hyperparameters, metrics.
6. Results: paper baseline, AM-FNO local baseline, KNO results.
7. Analysis: where KNO is stronger/weaker, especially rollout stability.
8. Limitations: data version differences, lack of DDP, metrics mismatch risk.
9. Reproduction commands.

## 14. Open Questions Before Full Training

1. NS-2D 主对比到底以 `v1e-4` 本地复现为主，还是也要把 `v1e-3` 纳入正式表格？
2. CFD-1D/CFD-2D 是否必须完全沿用 AM-FNO 的 `reduced_batch=5`，还是可以增加数据量来观察 KNO 是否受益？
3. KNO 结果报告是否优先用 `relative L2`，还是同时保留 KNO 原论文常用的 MSE/RMSE？
4. 是否需要把 AM-FNO 复现代码作为 git submodule，而不是只在实验指导中注明外部仓库？

## 15. Current Recommendation

第一版不要先改 DDP，也不要先跑大规模超参搜索。建议先完成：

1. `epochs=1` smoke test for all three datasets.
2. `epochs=100` medium run to check convergence.
3. `epochs=500` final run for one seed.
4. If CFD-2D underperforms, tune `o`, `m`, batch size, and variable normalization.

这样最容易判断问题出在 KNO 结构、数据接口、metric 口径，还是训练预算。
