import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

from src.models.main.models.deepswm import DeepSWM


def load_stats(stats_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    means_path = os.path.join(stats_dir, "means.npy")
    stds_path = os.path.join(stats_dir, "stds.npy")
    if not (os.path.exists(means_path) and os.path.exists(stds_path)):
        raise FileNotFoundError(f"Stats not found under: {stats_dir}")
    means = np.load(means_path)
    stds = np.load(stds_path)
    return means, stds


def list_h5_files_sorted(data_dir: str) -> Dict[datetime, str]:
    file_map: Dict[datetime, str] = {}
    for name in os.listdir(data_dir):
        if not name.endswith(".h5"):
            continue
        stem = os.path.splitext(name)[0]
        try:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        file_map[ts] = os.path.join(data_dir, name)
    return dict(sorted(file_map.items()))


def load_feature_vector(path: str, feature_dim: int) -> np.ndarray:
    try:
        with h5py.File(path, "r") as f:
            vec = f["features"][:]
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            if vec.ndim != 1:
                vec = vec.reshape(-1)
            if vec.shape[0] != feature_dim:
                tmp = np.zeros(feature_dim, dtype=np.float32)
                tmp[: min(feature_dim, vec.shape[0])] = vec[: min(feature_dim, vec.shape[0])]
                vec = tmp
            return vec.astype(np.float32)
    except Exception:
        return np.zeros(feature_dim, dtype=np.float32)


def build_feature_history(feature_dir: str, target_time: datetime, hours: int = 672, feature_dim: int = 128) -> Tuple[np.ndarray, int]:
    seq: List[np.ndarray] = []
    valid = 0
    for i in range(hours):
        t = target_time - timedelta(hours=i)
        fname = t.strftime("%Y%m%d_%H0000.h5")
        fpath = os.path.join(feature_dir, fname)
        if os.path.exists(fpath):
            vec = load_feature_vector(fpath, feature_dim)
            if np.any(vec != 0):
                valid += 1
            seq.append(vec)
        else:
            seq.append(np.zeros(feature_dim, dtype=np.float32))
    seq.reverse()
    arr = np.stack(seq, axis=0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32), valid


def build_image_history(file_map: Dict[datetime, str], target_time: datetime, history: int, channels: int = 10) -> Tuple[np.ndarray, int]:
    seq: List[np.ndarray] = []
    valid = 0
    times = [target_time - timedelta(hours=history - 1 - i) for i in range(history)]
    for t in times:
        p = file_map.get(t)
        if p is None:
            seq.append(np.zeros((channels, 256, 256), dtype=np.float32))
        else:
            with h5py.File(p, "r") as f:
                X = f["X"][:]
            seq.append(X.astype(np.float32))
            valid += 1
    arr = np.stack(seq, axis=0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, valid


def normalize_images(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (X - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-8)


def load_main_checkpoint(model: torch.nn.Module, ckpt_path: str, map_location: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state if isinstance(state, dict) else state
    filtered = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(filtered, strict=False)


def _ascend(path: str, levels: int) -> str:
    for _ in range(levels):
        path = os.path.dirname(path)
    return path


def run_main_inference(args) -> None:
    # Resolve roots
    this_file = os.path.abspath(__file__)
    ml_dir = _ascend(this_file, 5)  # <repo>/ml
    repo_root = os.path.dirname(ml_dir)

    # Resolve data_root (relative paths are repo-root based, like pretrain)
    if os.path.isabs(args.data_root):
        data_root = args.data_root
    else:
        data_root = os.path.normpath(os.path.join(repo_root, args.data_root))

    data_path = os.path.join(data_root, "all_data_hours")
    feature_dir = os.path.join(data_root, "all_features")

    # Stats under ml_dir/datasets/main/fold{fold}/train
    stats_train_dir = os.path.join(ml_dir, "datasets", "main", f"fold{args.fold}", "train")

    means, stds = load_stats(stats_train_dir)

    file_map = list_h5_files_sorted(data_path)
    if not file_map:
        raise RuntimeError("No valid .h5 files found for images")

    # Choose targets
    if getattr(args, "datetime", None):
        try:
            targets = [datetime.strptime(args.datetime, "%Y%m%d_%H%M%S")]
        except ValueError:
            raise ValueError("Invalid --datetime format, expected YYYYMMDD_HHMMSS")
    else:
        targets = list(file_map.keys())

    device = torch.device(f"cuda:{args.cuda_device}") if (args.cuda_device >= 0 and torch.cuda.is_available()) else torch.device("cpu")

    # Model params (match deepswm defaults used in training)
    model = DeepSWM(
        D=64,
        drop_path_rate=0.6,
        layer_scale_init_value=1.0e-6,
        L=128,
        L_SSE=3,
        L_LT=1,
        L_mixing=2,
        dropout_rates={
            "sse": 0.6,
            "dwcm": 0.6,
            "stssm": 0.6,
            "ltssm": 0.6,
            "mixing_ssm": 0.6,
            "head": 0.6,
        },
    ).to(device)

    ckpt = args.resume_from_checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(repo_root, ckpt)
    load_main_checkpoint(model, ckpt, device)
    model.eval()

    predictions: Dict[str, List[float]] = {}

    for target in targets:
        X_hist, _ = build_image_history(file_map, target, history=args.history, channels=means.shape[0])
        feats_hist, _ = build_feature_history(feature_dir, target, hours=672, feature_dim=128)

        X_norm = normalize_images(X_hist, means, stds)
        t1 = torch.from_numpy(X_norm).float().unsqueeze(0).to(device)
        t2 = torch.from_numpy(feats_hist).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(t1, t2)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

        key = target.strftime("%Y%m%d%H")
        predictions[key] = [round(p, 6) for p in probs]

    if getattr(args, "debug", False):
        for k, v in predictions.items():
            print(k, v)
        return

    out_dir = os.path.join(repo_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pred_24.json")
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions for {len(predictions)} timestamps to {out_path}") 
