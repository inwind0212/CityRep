from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .align import AlignedEmbedding, align_embedding
from .embeddings import PointEmbedding, RasterEmbedding, RegionEmbedding
from .io import write_json
from .metrics import classification_metrics, distribution_metrics, regression_metrics
from .protocols import load_protocol
from .splits import Split, make_splits
from .tasks import Task


class _ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str):
        self.X = torch.from_numpy(X).float()
        if task_type == "classification":
            self.y = torch.from_numpy(y.astype(np.int64))
        else:
            self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, name: str):
        super().__init__()
        if name == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif name == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        else:
            raise ValueError(f"Unsupported predictor: {name}")

    def forward(self, x):
        return self.net(x)


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    metrics_std: dict[str, float]
    per_seed: list[dict[str, Any]]
    save_dir: str


def evaluate(
    task: Task,
    embedding: RasterEmbedding | RegionEmbedding | PointEmbedding | AlignedEmbedding,
    *,
    protocol: str | dict[str, Any] = "block10_5seed_mlp1024",
    out_dir: str | Path = "results",
    normalize_embedding: bool = True,
) -> EvaluationResult:
    protocol_cfg = load_protocol(protocol) if isinstance(protocol, str) else dict(protocol)
    alignment_cfg = protocol_cfg.get("alignment", {})
    aligned = (
        embedding
        if isinstance(embedding, AlignedEmbedding)
        else align_embedding(
            task,
            embedding,
            normalize=bool(alignment_cfg.get("normalize", normalize_embedding)),
            fill_missing=alignment_cfg.get("fill_missing"),
        )
    )
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    aligned.save(save_dir / "aligned")

    split_cfg = protocol_cfg.get("split", {})
    splits = make_splits(task, split_cfg)
    predictor_cfg = protocol_cfg.get("predictor", {})
    per_seed: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    split_payload = {"protocol": protocol_cfg, "folds": []}

    seed_iter = tqdm(splits, desc=f"{task.task_id} seeds", unit="seed", leave=False, dynamic_ncols=True)
    for split in seed_iter:
        seed_iter.set_postfix(seed=split.seed)
        seed_dir = save_dir / f"seed{split.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics, predictions = _run_one_seed(aligned.X, task, split, predictor_cfg, seed_dir)
        per_seed.append({"seed": split.seed, **metrics})
        all_predictions.append(predictions)
        split_payload["folds"].append(
            {
                "seed": split.seed,
                "meta": split.meta,
                "train": task.samples.iloc[split.train_idx]["sample_id"].astype(str).tolist(),
                "val": task.samples.iloc[split.val_idx]["sample_id"].astype(str).tolist(),
                "test": task.samples.iloc[split.test_idx]["sample_id"].astype(str).tolist(),
            }
        )

    metrics_mean, metrics_std = _aggregate(per_seed)
    write_json(save_dir / "metrics.json", metrics_mean)
    write_json(save_dir / "metrics_std.json", metrics_std)
    write_json(save_dir / "metrics_per_seed.json", {"seeds": per_seed})
    write_json(save_dir / "split.json", split_payload)
    pd.concat(all_predictions, ignore_index=True).to_parquet(save_dir / "predictions.parquet", index=False)
    summary = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "n_samples": task.n_samples,
        "protocol_id": protocol_cfg.get("protocol_id"),
        "metrics": metrics_mean,
        "metrics_std": metrics_std,
        "save_dir": str(save_dir.resolve()),
    }
    write_json(save_dir / "run_summary.json", summary)
    return EvaluationResult(metrics_mean, metrics_std, per_seed, str(save_dir))


def _loss(task_type: str):
    if task_type == "regression":
        return lambda logits, y: F.mse_loss(logits, y)
    if task_type == "classification":
        return lambda logits, y: F.cross_entropy(logits, y)
    if task_type == "distribution":
        return lambda logits, y: F.kl_div(F.log_softmax(logits, dim=1), y, reduction="batchmean")
    raise ValueError(f"Unsupported task_type: {task_type}")


def _run_one_seed(X: np.ndarray, task: Task, split: Split, predictor_cfg: dict[str, Any], save_dir: Path) -> tuple[dict[str, float], pd.DataFrame]:
    seed = int(split.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(predictor_cfg.get("device", "cpu"))
    batch_size = int(predictor_cfg.get("batch_size", 512))
    epochs = int(predictor_cfg.get("epochs", 100))
    patience = int(predictor_cfg.get("patience", 10))
    hidden_dim = int(predictor_cfg.get("hidden_dim", 1024))
    lr = float(predictor_cfg.get("lr", 1e-3))
    predictor_name = str(predictor_cfg.get("name", "mlp")).lower()

    y = task.y
    out_dim = int(task.output_dim)
    model = _Head(int(X.shape[1]), out_dim, hidden_dim, predictor_name).to(device)
    loss_fn = _loss(task.task_type)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(_ArrayDataset(X[split.train_idx], y[split.train_idx], task.task_type), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_ArrayDataset(X[split.val_idx], y[split.val_idx], task.task_type), batch_size=batch_size, shuffle=False) if len(split.val_idx) else None
    test_loader = DataLoader(_ArrayDataset(X[split.test_idx], y[split.test_idx], task.task_type), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    wait = 0
    epoch_iter = tqdm(range(1, epochs + 1), desc=f"seed {seed} epochs", unit="epoch", leave=False, dynamic_ncols=True)
    for _epoch in epoch_iter:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, loss_fn, device)
            epoch_iter.set_postfix(best=f"{best_val:.4g}", val=f"{val_loss:.4g}", wait=wait)
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_dir / "model.pth")

    logits = _predict(model, test_loader, device)
    y_true = y[split.test_idx]
    if task.task_type == "regression":
        y_pred = logits
        metrics = regression_metrics(y_true, y_pred)
    elif task.task_type == "classification":
        y_pred = logits.argmax(axis=1)
        metrics = classification_metrics(y_true, logits)
    else:
        y_pred = _softmax_np(logits)
        metrics = distribution_metrics(y_true, y_pred)
    metrics["best_val"] = float(best_val) if np.isfinite(best_val) else float("nan")
    write_json(save_dir / "metrics.json", metrics)

    pred_df = pd.DataFrame({"sample_id": task.samples.iloc[split.test_idx]["sample_id"].astype(str).to_numpy(), "seed": split.seed})
    if task.task_type == "distribution":
        for i in range(y_true.shape[1]):
            pred_df[f"y_true_{i}"] = y_true[:, i]
            pred_df[f"y_pred_{i}"] = y_pred[:, i]
    else:
        pred_df["y_true"] = y_true.reshape(-1)
        pred_df["y_pred"] = y_pred.reshape(-1)
    return metrics, pred_df


def _eval_loss(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            total += float(loss_fn(model(xb.to(device)), yb.to(device)).item())
    return total / max(1, len(loader))


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for xb, _yb in loader:
            rows.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(rows)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _aggregate(per_seed: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted(set().union(*(row.keys() for row in per_seed)) - {"seed"})
    means = {}
    stds = {}
    for key in keys:
        vals = np.asarray([float(row[key]) for row in per_seed if key in row], dtype=np.float64)
        means[key] = float(vals.mean())
        stds[key] = float(vals.std(ddof=0))
    means["n_seeds"] = float(len(per_seed))
    return means, stds
