import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from timeit import default_timer

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class MatReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except NotImplementedError:
            self.data = h5py.File(self.file_path, "r")
            self.old_mat = False

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        return torch.from_numpy(x.astype(np.float32))


class CFDDatasetSingle(Dataset):
    def __init__(
        self,
        path,
        initial_step=10,
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        test_ratio=0.1,
        num_samples_max=-1,
    ):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            density = np.array(f["density"], dtype=np.float32)
            shape = density.shape
            if len(shape) == 3:
                self.data, self.grid = self._load_1d(f, density, reduced_batch, reduced_resolution_t, reduced_resolution)
            elif len(shape) == 4:
                self.data, self.grid = self._load_2d(f, density, reduced_batch, reduced_resolution_t, reduced_resolution)
            else:
                raise ValueError(f"Only 1D/2D CFD files are supported, got density shape {shape}")

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if test_idx <= 0:
            raise ValueError("test_ratio creates an empty test split")
        self.data = self.data[:test_idx] if if_test else self.data[test_idx:num_samples_max]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.initial_step = initial_step

    @staticmethod
    def _load_1d(f, density, reduced_batch, reduced_resolution_t, reduced_resolution):
        n, t, x = density.shape
        data = np.zeros(
            [n // reduced_batch, x // reduced_resolution, math.ceil(t / reduced_resolution_t), 3],
            dtype=np.float32,
        )
        for idx, key in enumerate(["density", "pressure", "Vx"]):
            arr = np.array(f[key], dtype=np.float32)
            arr = arr[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
            data[..., idx] = np.transpose(arr, (0, 2, 1))
        grid = torch.tensor(np.array(f["x-coordinate"], dtype=np.float32)[::reduced_resolution], dtype=torch.float32)
        grid = grid.unsqueeze(-1)
        return data, grid

    @staticmethod
    def _load_2d(f, density, reduced_batch, reduced_resolution_t, reduced_resolution):
        n, t, x, y = density.shape
        data = np.zeros(
            [
                n // reduced_batch,
                x // reduced_resolution,
                y // reduced_resolution,
                math.ceil(t / reduced_resolution_t),
                4,
            ],
            dtype=np.float32,
        )
        for idx, key in enumerate(["density", "pressure", "Vx", "Vy"]):
            arr = np.array(f[key], dtype=np.float32)
            arr = arr[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution]
            data[..., idx] = np.transpose(arr, (0, 2, 3, 1))
        x_coord = torch.tensor(np.array(f["x-coordinate"], dtype=np.float32), dtype=torch.float32)
        y_coord = torch.tensor(np.array(f["y-coordinate"], dtype=np.float32), dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coord, y_coord, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[::reduced_resolution, ::reduced_resolution]
        return data, grid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ..., : self.initial_step, :], self.data[idx], self.grid


class LinearEncoder(nn.Module):
    def __init__(self, in_dim, op_size):
        super().__init__()
        self.layer = nn.Linear(in_dim, op_size)

    def forward(self, x):
        return self.layer(x)


class LinearDecoder(nn.Module):
    def __init__(self, out_dim, op_size):
        super().__init__()
        self.layer = nn.Linear(op_size, out_dim)

    def forward(self, x):
        return self.layer(x)


class KoopmanOperator1D(nn.Module):
    def __init__(self, op_size, modes_x=16):
        super().__init__()
        self.modes_x = modes_x
        scale = 1 / (op_size * op_size)
        self.koopman_matrix = nn.Parameter(scale * torch.rand(op_size, op_size, modes_x, dtype=torch.cfloat))

    @staticmethod
    def time_marching(x, weights):
        return torch.einsum("btx,tfx->bfx", x, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes_x, x_ft.shape[-1])
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes] = self.time_marching(x_ft[:, :, :modes], self.koopman_matrix[:, :, :modes])
        return torch.fft.irfft(out_ft, n=x.size(-1))


class KoopmanOperator2D(nn.Module):
    def __init__(self, op_size, modes_x=16, modes_y=16):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1 / (op_size * op_size)
        self.koopman_matrix = nn.Parameter(
            scale * torch.rand(op_size, op_size, modes_x, modes_y, dtype=torch.cfloat)
        )

    @staticmethod
    def time_marching(x, weights):
        return torch.einsum("btxy,tfxy->bfxy", x, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft2(x)
        modes_x = min(self.modes_x, x_ft.shape[-2])
        modes_y = min(self.modes_y, x_ft.shape[-1])
        weights = self.koopman_matrix[:, :, :modes_x, :modes_y]
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes_x, :modes_y] = self.time_marching(x_ft[:, :, :modes_x, :modes_y], weights)
        out_ft[:, :, -modes_x:, :modes_y] = self.time_marching(x_ft[:, :, -modes_x:, :modes_y], weights)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class KNO1dFlex(nn.Module):
    def __init__(self, in_dim, out_dim, op_size=32, modes=16, decompose=8):
        super().__init__()
        self.decompose = decompose
        self.enc = LinearEncoder(in_dim, op_size)
        self.dec_pred = LinearDecoder(out_dim, op_size)
        self.dec_recon = LinearDecoder(in_dim, op_size)
        self.koopman_layer = KoopmanOperator1D(op_size, modes_x=modes)
        self.w0 = nn.Conv1d(op_size, op_size, 1)

    def forward(self, x):
        z_recon = torch.tanh(self.enc(x))
        x_recon = self.dec_recon(z_recon)

        z = torch.tanh(self.enc(x)).permute(0, 2, 1)
        z_w = z
        for _ in range(self.decompose):
            z = z + self.koopman_layer(z)
        z = torch.tanh(self.w0(z_w) + z).permute(0, 2, 1)
        return self.dec_pred(z), x_recon


class KNO2dFlex(nn.Module):
    def __init__(self, in_dim, out_dim, op_size=32, modes=16, decompose=8):
        super().__init__()
        self.decompose = decompose
        self.enc = LinearEncoder(in_dim, op_size)
        self.dec_pred = LinearDecoder(out_dim, op_size)
        self.dec_recon = LinearDecoder(in_dim, op_size)
        self.koopman_layer = KoopmanOperator2D(op_size, modes_x=modes, modes_y=modes)
        self.w0 = nn.Conv2d(op_size, op_size, 1)

    def forward(self, x):
        z_recon = torch.tanh(self.enc(x))
        x_recon = self.dec_recon(z_recon)

        z = torch.tanh(self.enc(x)).permute(0, 3, 1, 2)
        z_w = z
        for _ in range(self.decompose):
            z = z + self.koopman_layer(z)
        z = torch.tanh(self.w0(z_w) + z).permute(0, 2, 3, 1)
        return self.dec_pred(z), x_recon


class RelativeLpLoss:
    def __init__(self, p=2, eps=1e-12):
        self.p = p
        self.eps = eps

    def __call__(self, x, y):
        batch = x.shape[0]
        diff = torch.norm(x.reshape(batch, -1) - y.reshape(batch, -1), p=self.p, dim=1)
        denom = torch.norm(y.reshape(batch, -1), p=self.p, dim=1).clamp_min(self.eps)
        return torch.sum(diff / denom)


class CsvLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def count_params(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if p.is_complex() else 1)
    return total


def parse_args():
    parser = argparse.ArgumentParser("Train KNO on AM-FNO datasets")
    parser.add_argument("--benchmark", choices=["ns2d", "cfd1d", "cfd2d"], required=True)
    parser.add_argument("--ns-file", type=Path, default=None)
    parser.add_argument("--cfd-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--o", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--initial-step", type=int, default=10)
    parser.add_argument("--t-train", type=int, default=21)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--train-downsample", type=int, default=1)
    parser.add_argument("--test-downsample", type=int, default=1)
    parser.add_argument("--reduced-resolution", type=int, default=None)
    parser.add_argument("--reduced-resolution-t", type=int, default=None)
    parser.add_argument("--reduced-batch", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    return parser.parse_args()


def make_output_dir(args):
    run_name = args.run_name or f"kno_{args.benchmark}_seed{args.seed}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(path, obj):
    serializable = {k: str(v) if isinstance(v, Path) else v for k, v in obj.items()}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


def write_env(out_dir, params):
    with (out_dir / "env.txt").open("w", encoding="utf-8") as f:
        f.write(f"python: {sys.version}\n")
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"cuda: {torch.version.cuda}\n")
        f.write(f"cuda_available: {torch.cuda.is_available()}\n")
        f.write(f"visible_gpu_count: {torch.cuda.device_count()}\n")
        if torch.cuda.is_available():
            f.write(f"visible_device_0: {torch.cuda.get_device_name(0)}\n")
        f.write(f"params_count_complex_as_2: {params}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")


def require_path(name, path):
    if path is None:
        raise ValueError(f"{name} is required")
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return Path(path)


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_ns_loaders(args):
    ns_file = require_path("--ns-file", args.ns_file)
    data = MatReader(str(ns_file)).read_field("u")
    if data.ndim != 4:
        raise ValueError(f"Expected NS field u with shape [N,X,Y,T], got {tuple(data.shape)}")
    ntrain = args.ntrain or 1000
    ntest = args.ntest or 200
    train_sub = args.train_downsample
    test_sub = args.test_downsample
    train_a = data[:ntrain, ::train_sub, ::train_sub, : args.t_in]
    train_u = data[:ntrain, ::train_sub, ::train_sub, args.t_in : args.t_in + args.t_out]
    test_a = data[-ntest:, ::test_sub, ::test_sub, : args.t_in]
    test_u = data[-ntest:, ::test_sub, ::test_sub, args.t_in : args.t_in + args.t_out]
    batch_size = args.batch_size or 10
    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    model = KNO2dFlex(args.t_in, 1, op_size=args.o, modes=args.modes, decompose=args.decompose)
    meta = {"ntrain": ntrain, "ntest": ntest, "t_out": args.t_out, "kind": "ns2d"}
    print(f"NS data loaded: train {tuple(train_a.shape)} -> {tuple(train_u.shape)}, test {tuple(test_a.shape)}")
    return model, train_loader, test_loader, meta


def build_cfd_loaders(args, dim):
    cfd_file = require_path("--cfd-file", args.cfd_file)
    reduced_resolution = args.reduced_resolution if args.reduced_resolution is not None else (8 if dim == 1 else 2)
    reduced_resolution_t = args.reduced_resolution_t if args.reduced_resolution_t is not None else (5 if dim == 1 else 1)
    train_data = CFDDatasetSingle(
        cfd_file,
        initial_step=args.initial_step,
        reduced_resolution=reduced_resolution,
        reduced_resolution_t=reduced_resolution_t,
        reduced_batch=args.reduced_batch,
        test_ratio=args.test_ratio,
    )
    test_data = CFDDatasetSingle(
        cfd_file,
        initial_step=args.initial_step,
        reduced_resolution=reduced_resolution,
        reduced_resolution_t=reduced_resolution_t,
        reduced_batch=args.reduced_batch,
        if_test=True,
        test_ratio=args.test_ratio,
    )
    batch_size = args.batch_size or (32 if dim == 1 else 8)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    variables = 3 if dim == 1 else 4
    in_dim = args.initial_step * variables
    cls = KNO1dFlex if dim == 1 else KNO2dFlex
    model = cls(in_dim, variables, op_size=args.o, modes=args.modes, decompose=args.decompose)
    meta = {
        "ntrain": len(train_data),
        "ntest": len(test_data),
        "t_out": args.t_train - args.initial_step,
        "variables": variables,
        "kind": f"cfd{dim}d",
    }
    first_x, first_y, first_grid = train_data[0]
    print(f"CFD-{dim}D data loaded: x {tuple(first_x.shape)}, y {tuple(first_y.shape)}, grid {tuple(first_grid.shape)}")
    return model, train_loader, test_loader, meta


def train_ns(model, train_loader, test_loader, args, device, out_dir, meta):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    mse = nn.MSELoss()
    rel = RelativeLpLoss()
    logger = CsvLogger(
        out_dir / "metrics.csv",
        [
            "epoch",
            "seconds",
            "train_step_rel_l2",
            "train_full_rel_l2",
            "test_step_rel_l2",
            "test_full_rel_l2",
            "train_pred_mse",
            "train_recon_mse",
            "test_pred_mse",
            "test_recon_mse",
            "lr",
        ],
    )
    best = float("inf")
    try:
        for ep in range(args.epochs):
            model.train()
            t1 = default_timer()
            train_step_rel = train_full_rel = train_pred_mse = train_recon_mse = 0.0
            for xx, yy in train_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                inp = xx
                pred = None
                pred_mse = recon_mse = 0.0
                step_rel = 0.0
                for t in range(meta["t_out"]):
                    y = yy[..., t : t + 1]
                    im, recon = model(inp)
                    pred_mse = pred_mse + mse(im, y)
                    recon_mse = recon_mse + mse(recon, inp)
                    step_rel = step_rel + rel(im, y)
                    pred = im if pred is None else torch.cat((pred, im), dim=-1)
                    inp = torch.cat((inp[..., 1:], im), dim=-1)
                loss = 5.0 * pred_mse + 0.5 * recon_mse
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_step_rel += step_rel.item()
                train_full_rel += rel(pred, yy).item()
                train_pred_mse += pred_mse.item()
                train_recon_mse += recon_mse.item() / meta["t_out"]

            test_stats = eval_ns(model, test_loader, device, meta, mse, rel)
            scheduler.step()
            seconds = default_timer() - t1
            row = {
                "epoch": ep,
                "seconds": f"{seconds:.6f}",
                "train_step_rel_l2": f"{train_step_rel / meta['ntrain'] / meta['t_out']:.8e}",
                "train_full_rel_l2": f"{train_full_rel / meta['ntrain']:.8e}",
                "test_step_rel_l2": f"{test_stats['step_rel'] / meta['ntest'] / meta['t_out']:.8e}",
                "test_full_rel_l2": f"{test_stats['full_rel'] / meta['ntest']:.8e}",
                "train_pred_mse": f"{train_pred_mse / len(train_loader):.8e}",
                "train_recon_mse": f"{train_recon_mse / len(train_loader):.8e}",
                "test_pred_mse": f"{test_stats['pred_mse'] / len(test_loader):.8e}",
                "test_recon_mse": f"{test_stats['recon_mse'] / len(test_loader):.8e}",
                "lr": f"{scheduler.get_last_lr()[0]:.8e}",
            }
            logger.write(row)
            score = float(row["test_full_rel_l2"])
            if args.save_checkpoint and score < best:
                best = score
                torch.save(model.state_dict(), out_dir / "checkpoint_best.pt")
            if ep % args.log_every == 0:
                print(
                    f"epoch {ep:04d} | {seconds:.2f}s | "
                    f"train_full {float(row['train_full_rel_l2']):.6e} | "
                    f"test_full {float(row['test_full_rel_l2']):.6e}"
                )
    finally:
        logger.close()
    if args.save_checkpoint:
        torch.save(model.state_dict(), out_dir / "checkpoint_last.pt")


def eval_ns(model, loader, device, meta, mse, rel):
    model.eval()
    stats = {"step_rel": 0.0, "full_rel": 0.0, "pred_mse": 0.0, "recon_mse": 0.0}
    with torch.no_grad():
        for xx, yy in loader:
            xx = xx.to(device)
            yy = yy.to(device)
            inp = xx
            pred = None
            pred_mse = recon_mse = 0.0
            for t in range(meta["t_out"]):
                y = yy[..., t : t + 1]
                im, recon = model(inp)
                pred_mse = pred_mse + mse(im, y)
                recon_mse = recon_mse + mse(recon, inp)
                stats["step_rel"] += rel(im, y).item()
                pred = im if pred is None else torch.cat((pred, im), dim=-1)
                inp = torch.cat((inp[..., 1:], im), dim=-1)
            stats["full_rel"] += rel(pred, yy).item()
            stats["pred_mse"] += pred_mse.item()
            stats["recon_mse"] += recon_mse.item() / meta["t_out"]
    return stats


def train_cfd(model, train_loader, test_loader, args, device, out_dir, meta):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    mse = nn.MSELoss()
    rel = RelativeLpLoss()
    logger = CsvLogger(
        out_dir / "metrics.csv",
        [
            "epoch",
            "seconds",
            "train_step_rel_l2",
            "train_full_rel_l2",
            "test_step_rel_l2",
            "test_full_rel_l2",
            "train_pred_mse",
            "train_recon_mse",
            "test_pred_mse",
            "test_recon_mse",
            "lr",
        ],
    )
    best = float("inf")
    try:
        for ep in range(args.epochs):
            model.train()
            t1 = default_timer()
            train_step_rel = train_full_rel = train_pred_mse = train_recon_mse = 0.0
            for xx, yy, _grid in train_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                pred, pred_mse, recon_mse, step_rel = cfd_rollout(model, xx, yy, args, meta, mse, rel, train=True)
                loss = 5.0 * pred_mse + 0.5 * recon_mse
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                target = yy[..., : args.t_train, :]
                train_step_rel += step_rel.item()
                train_full_rel += rel(pred, target).item()
                train_pred_mse += pred_mse.item()
                train_recon_mse += recon_mse.item() / meta["t_out"]

            test_stats = eval_cfd(model, test_loader, args, device, meta, mse, rel)
            scheduler.step()
            seconds = default_timer() - t1
            row = {
                "epoch": ep,
                "seconds": f"{seconds:.6f}",
                "train_step_rel_l2": f"{train_step_rel / meta['ntrain'] / meta['t_out']:.8e}",
                "train_full_rel_l2": f"{train_full_rel / meta['ntrain']:.8e}",
                "test_step_rel_l2": f"{test_stats['step_rel'] / meta['ntest'] / meta['t_out']:.8e}",
                "test_full_rel_l2": f"{test_stats['full_rel'] / meta['ntest']:.8e}",
                "train_pred_mse": f"{train_pred_mse / len(train_loader):.8e}",
                "train_recon_mse": f"{train_recon_mse / len(train_loader):.8e}",
                "test_pred_mse": f"{test_stats['pred_mse'] / len(test_loader):.8e}",
                "test_recon_mse": f"{test_stats['recon_mse'] / len(test_loader):.8e}",
                "lr": f"{scheduler.get_last_lr()[0]:.8e}",
            }
            logger.write(row)
            score = float(row["test_full_rel_l2"])
            if args.save_checkpoint and score < best:
                best = score
                torch.save(model.state_dict(), out_dir / "checkpoint_best.pt")
            if ep % args.log_every == 0:
                print(
                    f"epoch {ep:04d} | {seconds:.2f}s | "
                    f"train_full {float(row['train_full_rel_l2']):.6e} | "
                    f"test_full {float(row['test_full_rel_l2']):.6e}"
                )
    finally:
        logger.close()
    if args.save_checkpoint:
        torch.save(model.state_dict(), out_dir / "checkpoint_last.pt")


def cfd_rollout(model, xx, yy, args, meta, mse, rel, train):
    pred = yy[..., : args.initial_step, :]
    pred_mse = recon_mse = 0.0
    step_rel = 0.0
    inp = xx
    inp_shape = list(inp.shape[:-2]) + [-1]
    for t in range(args.initial_step, args.t_train):
        model_in = inp.reshape(inp_shape)
        y = yy[..., t : t + 1, :]
        im, recon = model(model_in)
        im = im.unsqueeze(-2)
        pred_mse = pred_mse + mse(im, y)
        recon_mse = recon_mse + mse(recon, model_in)
        step_rel = step_rel + rel(im, y)
        pred = torch.cat((pred, im), dim=-2)
        inp = torch.cat((inp[..., 1:, :], im), dim=-2)
    return pred, pred_mse, recon_mse, step_rel


def eval_cfd(model, loader, args, device, meta, mse, rel):
    model.eval()
    stats = {"step_rel": 0.0, "full_rel": 0.0, "pred_mse": 0.0, "recon_mse": 0.0}
    with torch.no_grad():
        for xx, yy, _grid in loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred, pred_mse, recon_mse, step_rel = cfd_rollout(model, xx, yy, args, meta, mse, rel, train=False)
            pred_window = pred[..., args.initial_step : args.t_train, :]
            target_window = yy[..., args.initial_step : args.t_train, :]
            stats["step_rel"] += step_rel.item()
            stats["full_rel"] += rel(pred_window, target_window).item()
            stats["pred_mse"] += pred_mse.item()
            stats["recon_mse"] += recon_mse.item() / meta["t_out"]
    return stats


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    out_dir = make_output_dir(args)
    write_json(out_dir / "args.json", vars(args))

    if args.benchmark == "ns2d":
        model, train_loader, test_loader, meta = build_ns_loaders(args)
        params = count_params(model)
        print(f"parameters: {params}")
        write_env(out_dir, params)
        train_ns(model, train_loader, test_loader, args, device, out_dir, meta)
    elif args.benchmark == "cfd1d":
        model, train_loader, test_loader, meta = build_cfd_loaders(args, dim=1)
        params = count_params(model)
        print(f"parameters: {params}")
        write_env(out_dir, params)
        train_cfd(model, train_loader, test_loader, args, device, out_dir, meta)
    elif args.benchmark == "cfd2d":
        model, train_loader, test_loader, meta = build_cfd_loaders(args, dim=2)
        params = count_params(model)
        print(f"parameters: {params}")
        write_env(out_dir, params)
        train_cfd(model, train_loader, test_loader, args, device, out_dir, meta)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
