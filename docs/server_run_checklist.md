# Server Run Checklist: KNO on AM-FNO Data

本文档假设当前服务器较忙，只有物理 GPU `cuda=1` 空闲。因此第一阶段全部命令都以单卡 `CUDA_VISIBLE_DEVICES=1` 为前提。等服务器空闲后，再切换为多 GPU 并行跑不同实验。

> 注意：当前仓库第一版已经完成实验设计和目录规范；正式训练入口 `experiments/train_kno_amfno.py` 会在下一步实现。本文档先固定服务器操作流程和命令模板。

## 1. Clone And Update Repo

```bash
git clone git@github.com:Quin557/kno-experiments.git
cd kno-experiments
git pull origin main
```

如果仓库已经在服务器上：

```bash
cd /path/to/kno-experiments
git pull origin main
```

## 2. Check GPU Status

先确认 `cuda=1` 是否真的空闲：

```bash
nvidia-smi
```

只看第 1 号卡：

```bash
nvidia-smi -i 1
```

后续所有单卡实验都加：

```bash
export CUDA_VISIBLE_DEVICES=1
```

设置后，程序内部看到的 `cuda:0` 实际对应物理 GPU 1。这一点很重要：如果命令里设置了 `CUDA_VISIBLE_DEVICES=1`，Python 里仍然应使用 `cuda` 或 `cuda:0`，不要写 `cuda:1`。

## 3. Prepare External Code

KNO 官方仓库只作为依赖和参考，不直接提交到本实验仓库。

```bash
mkdir -p external

if [ ! -d external/KoopmanLab ]; then
  git clone https://github.com/Koopman-Laboratory/KoopmanLab external/KoopmanLab
fi

if [ ! -d external/am_fno_repro ]; then
  git clone https://github.com/Quin557/am_fno_repro external/am_fno_repro
fi
```

记录外部代码版本：

```bash
git -C external/KoopmanLab rev-parse HEAD
git -C external/am_fno_repro rev-parse HEAD
```

建议把输出复制到最终实验记录里。

## 4. Python Environment

### 4.1 First inspect existing environments

```bash
conda env list
```

如果服务器已有合适环境，比如 `torch121`：

```bash
conda activate torch121
python - <<'PY'
import torch, numpy, scipy, h5py
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
PY
```

### 4.2 If no suitable environment exists

```bash
conda create -n kno-amfno python=3.10 -y
conda activate kno-amfno

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy h5py matplotlib tqdm pandas pyyaml
pip install -e external/KoopmanLab
```

再次验证：

```bash
export CUDA_VISIBLE_DEVICES=1
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("visible device 0 name:", torch.cuda.get_device_name(0))
PY
```

## 5. Data Placement

推荐服务器数据结构：

```text
/path/to/amfno_data/
  cfd/
    1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5
    2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5
  ns2d/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
```

创建私有路径配置：

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
```

示例内容：

```bash
DATA_ROOT=/path/to/amfno_data

CFD1D_FILE=cfd/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5
CFD2D_FILE=cfd/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5

NS2D_V1E3=ns2d/ns_V1e-3_N5000_T50.mat
NS2D_V1E4=ns2d/ns_V1e-4_N10000_T30.mat
```

快速检查文件是否存在：

```bash
source configs/data_paths.env

ls -lh "$DATA_ROOT/$CFD1D_FILE"
ls -lh "$DATA_ROOT/$CFD2D_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3"
ls -lh "$DATA_ROOT/$NS2D_V1E4"
```

## 6. Single-GPU Plan While Only cuda=1 Is Free

当前只使用物理 GPU 1。建议按下面顺序跑，因为风险从低到高：

1. NS-2D v1e-4 smoke test。
2. CFD-1D smoke test。
3. CFD-2D smoke test。
4. NS-2D v1e-4 medium run。
5. CFD-1D medium run。
6. CFD-2D medium run。
7. Full runs。

### 6.1 Smoke tests

Smoke test 只跑 1 epoch，目的不是看结果，而是确认：

- 数据能读取；
- shape 对齐；
- forward/backward 正常；
- metrics.csv 能写出；
- 显存不会炸。

命令模板：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

python experiments/train_kno_amfno.py \
  --benchmark ns2d \
  --ns-file "$DATA_ROOT/$NS2D_V1E4" \
  --run-name smoke_ns2d_v1e4_cuda1 \
  --epochs 1 \
  --batch-size 10 \
  --device cuda \
  --output-dir outputs
```

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

python experiments/train_kno_amfno.py \
  --benchmark cfd1d \
  --cfd-file "$DATA_ROOT/$CFD1D_FILE" \
  --run-name smoke_cfd1d_cuda1 \
  --epochs 1 \
  --batch-size 32 \
  --device cuda \
  --output-dir outputs \
  --reduced-resolution 8 \
  --reduced-resolution-t 5 \
  --reduced-batch 5
```

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

python experiments/train_kno_amfno.py \
  --benchmark cfd2d \
  --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name smoke_cfd2d_cuda1 \
  --epochs 1 \
  --batch-size 8 \
  --device cuda \
  --output-dir outputs \
  --reduced-resolution 2 \
  --reduced-resolution-t 1 \
  --reduced-batch 5
```

### 6.2 Medium runs

Medium run 先跑 100 epoch，用来判断 KNO 是否收敛。建议开 `tmux`，防止 SSH 断开。

```bash
tmux new -s kno_cuda1
```

NS-2D v1e-4：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

python experiments/train_kno_amfno.py \
  --benchmark ns2d \
  --ns-file "$DATA_ROOT/$NS2D_V1E4" \
  --run-name kno_ns2d_v1e4_o32_m16_r8_ep100_cuda1 \
  --epochs 100 \
  --batch-size 10 \
  --o 32 \
  --modes 16 \
  --decompose 8 \
  --t-in 10 \
  --t-out 10 \
  --device cuda \
  --output-dir outputs \
  2>&1 | tee logs/kno_ns2d_v1e4_ep100_cuda1.log
```

CFD-1D：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

python experiments/train_kno_amfno.py \
  --benchmark cfd1d \
  --cfd-file "$DATA_ROOT/$CFD1D_FILE" \
  --run-name kno_cfd1d_o32_m16_r8_ep100_cuda1 \
  --epochs 100 \
  --batch-size 32 \
  --o 32 \
  --modes 16 \
  --decompose 8 \
  --initial-step 10 \
  --t-train 21 \
  --reduced-resolution 8 \
  --reduced-resolution-t 5 \
  --reduced-batch 5 \
  --device cuda \
  --output-dir outputs \
  2>&1 | tee logs/kno_cfd1d_ep100_cuda1.log
```

CFD-2D：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

python experiments/train_kno_amfno.py \
  --benchmark cfd2d \
  --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name kno_cfd2d_o32_m16_r8_ep100_cuda1 \
  --epochs 100 \
  --batch-size 8 \
  --o 32 \
  --modes 16 \
  --decompose 8 \
  --initial-step 10 \
  --t-train 21 \
  --reduced-resolution 2 \
  --reduced-resolution-t 1 \
  --reduced-batch 5 \
  --device cuda \
  --output-dir outputs \
  2>&1 | tee logs/kno_cfd2d_ep100_cuda1.log
```

### 6.3 Full runs on cuda=1

如果 100 epoch 正常收敛，再跑 500 epoch。单卡情况下建议一次只跑一个 full run。

优先级：

1. `cfd1d`: 最可能较快得到可比结果。
2. `ns2d_v1e4`: 和已有 AM-FNO 复现结果最容易对齐。
3. `cfd2d`: 计算更重，最后跑。
4. `ns2d_v1e3`: 作为 KNO 原论文风格补充。

CFD-1D full：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

python experiments/train_kno_amfno.py \
  --benchmark cfd1d \
  --cfd-file "$DATA_ROOT/$CFD1D_FILE" \
  --run-name kno_cfd1d_o32_m16_r8_ep500_seed42 \
  --epochs 500 \
  --batch-size 32 \
  --seed 42 \
  --o 32 \
  --modes 16 \
  --decompose 8 \
  --initial-step 10 \
  --t-train 21 \
  --reduced-resolution 8 \
  --reduced-resolution-t 5 \
  --reduced-batch 5 \
  --save-checkpoint \
  --device cuda \
  --output-dir outputs \
  2>&1 | tee logs/kno_cfd1d_ep500_seed42.log
```

NS-2D v1e-4 full：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

python experiments/train_kno_amfno.py \
  --benchmark ns2d \
  --ns-file "$DATA_ROOT/$NS2D_V1E4" \
  --run-name kno_ns2d_v1e4_o32_m16_r8_ep500_seed42 \
  --epochs 500 \
  --batch-size 10 \
  --seed 42 \
  --o 32 \
  --modes 16 \
  --decompose 8 \
  --t-in 10 \
  --t-out 10 \
  --save-checkpoint \
  --device cuda \
  --output-dir outputs \
  2>&1 | tee logs/kno_ns2d_v1e4_ep500_seed42.log
```

CFD-2D full：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

python experiments/train_kno_amfno.py \
  --benchmark cfd2d \
  --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name kno_cfd2d_o32_m16_r8_ep500_seed42 \
  --epochs 500 \
  --batch-size 8 \
  --seed 42 \
  --o 32 \
  --modes 16 \
  --decompose 8 \
  --initial-step 10 \
  --t-train 21 \
  --reduced-resolution 2 \
  --reduced-resolution-t 1 \
  --reduced-batch 5 \
  --save-checkpoint \
  --device cuda \
  --output-dir outputs \
  2>&1 | tee logs/kno_cfd2d_ep500_seed42.log
```

## 7. Monitoring

查看 GPU：

```bash
watch -n 5 nvidia-smi
```

查看日志：

```bash
tail -f logs/kno_cfd1d_ep500_seed42.log
```

查看指标：

```bash
tail -n 5 outputs/kno_cfd1d_o32_m16_r8_ep500_seed42/metrics.csv
```

如果显存不足：

1. 先降低 batch size。
2. CFD-2D 可先用 `--batch-size 4`。
3. 不优先降低 `modes`，因为 modes 会影响 KNO 的频域表示能力。

## 8. When The Server Becomes Idle

当多张 GPU 空闲时，不必立刻改 DDP。第一阶段最稳的方式是“多进程多实验”，一张卡跑一个实验。

### 8.1 Four main experiments in parallel

```bash
source configs/data_paths.env
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python experiments/train_kno_amfno.py \
  --benchmark ns2d --ns-file "$DATA_ROOT/$NS2D_V1E4" \
  --run-name kno_ns2d_v1e4_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 10 --seed 42 --o 32 --modes 16 --decompose 8 \
  --t-in 10 --t-out 10 --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_ns2d_v1e4_gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python experiments/train_kno_amfno.py \
  --benchmark ns2d --ns-file "$DATA_ROOT/$NS2D_V1E3" \
  --run-name kno_ns2d_v1e3_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 10 --seed 42 --o 32 --modes 16 --decompose 8 \
  --t-in 10 --t-out 10 --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_ns2d_v1e3_gpu1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python experiments/train_kno_amfno.py \
  --benchmark cfd1d --cfd-file "$DATA_ROOT/$CFD1D_FILE" \
  --run-name kno_cfd1d_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 32 --seed 42 --o 32 --modes 16 --decompose 8 \
  --initial-step 10 --t-train 21 \
  --reduced-resolution 8 --reduced-resolution-t 5 --reduced-batch 5 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_cfd1d_gpu2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python experiments/train_kno_amfno.py \
  --benchmark cfd2d --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name kno_cfd2d_o32_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 8 --seed 42 --o 32 --modes 16 --decompose 8 \
  --initial-step 10 --t-train 21 \
  --reduced-resolution 2 --reduced-resolution-t 1 --reduced-batch 5 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_cfd2d_gpu3.log 2>&1 &

jobs
```

### 8.2 Optional tuning jobs

如果主实验都跑完，再用空闲 GPU 跑小规模调参：

```bash
source configs/data_paths.env
mkdir -p logs

CUDA_VISIBLE_DEVICES=4 python experiments/train_kno_amfno.py \
  --benchmark cfd2d --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name kno_cfd2d_o64_m16_r8_ep500_seed42 \
  --epochs 500 --batch-size 8 --seed 42 --o 64 --modes 16 --decompose 8 \
  --initial-step 10 --t-train 21 \
  --reduced-resolution 2 --reduced-resolution-t 1 --reduced-batch 5 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_cfd2d_o64_gpu4.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python experiments/train_kno_amfno.py \
  --benchmark cfd2d --cfd-file "$DATA_ROOT/$CFD2D_FILE" \
  --run-name kno_cfd2d_o32_m24_r8_ep500_seed42 \
  --epochs 500 --batch-size 8 --seed 42 --o 32 --modes 24 --decompose 8 \
  --initial-step 10 --t-train 21 \
  --reduced-resolution 2 --reduced-resolution-t 1 --reduced-batch 5 \
  --save-checkpoint --device cuda --output-dir outputs \
  > logs/kno_cfd2d_m24_gpu5.log 2>&1 &
```

## 9. After A Run Finishes

生成轻量结果摘要，后续提交到 git：

```bash
mkdir -p results

python - <<'PY'
from pathlib import Path
import csv

rows = []
for metric_path in Path("outputs").glob("*/metrics.csv"):
    with metric_path.open(newline="") as f:
        data = list(csv.DictReader(f))
    if not data:
        continue
    last = data[-1]
    best = min(data, key=lambda r: float(r.get("test_full_rel_l2") or r.get("test_rel_l2") or "inf"))
    rows.append({
        "run": metric_path.parent.name,
        "last_epoch": last.get("epoch", ""),
        "last_test_step_rel_l2": last.get("test_step_rel_l2", ""),
        "last_test_full_rel_l2": last.get("test_full_rel_l2", last.get("test_rel_l2", "")),
        "best_epoch": best.get("epoch", ""),
        "best_test_step_rel_l2": best.get("test_step_rel_l2", ""),
        "best_test_full_rel_l2": best.get("test_full_rel_l2", best.get("test_rel_l2", "")),
    })

out = Path("results/kno_run_summary.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "run", "last_epoch", "last_test_step_rel_l2", "last_test_full_rel_l2",
        "best_epoch", "best_test_step_rel_l2", "best_test_full_rel_l2"
    ])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

提交轻量结果：

```bash
git status --short
git add results/kno_run_summary.csv reports/
git commit -m "Add KNO experiment results"
git push origin main
```

不要提交：

- `data/`
- `outputs/*/*.pt`
- checkpoints
- 大日志
- 原始 `.mat/.hdf5/.npy`

## 10. Recommended Current Action

在只有 `cuda=1` 空闲时，建议当前先做：

```bash
export CUDA_VISIBLE_DEVICES=1
nvidia-smi -i 1
```

等下一步训练脚本提交后，先按顺序跑：

```bash
# 1. NS-2D smoke
# 2. CFD-1D smoke
# 3. CFD-2D smoke
# 4. CFD-1D 100 epoch
# 5. NS-2D v1e-4 100 epoch
# 6. CFD-2D 100 epoch
```

如果三组 100 epoch 都稳定，再进入 500 epoch 正式实验。
