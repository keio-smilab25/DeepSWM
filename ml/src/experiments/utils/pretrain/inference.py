import os
import re
import yaml
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import h5py

from src.models.pretrain.sparsemae import (
    mae_vit_base_patch16_dec512d8b,
    vit_for_FT64d4b,
    vit_for_FT32d4b,
    vit_for_FT128db,
    SparseMAE,
)


def load_stats(stats_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load normalization statistics (means/stds) from the training fold cache."""
    means_path = os.path.join(stats_dir, "means.npy")
    stds_path = os.path.join(stats_dir, "stds.npy")
    if not (os.path.exists(means_path) and os.path.exists(stds_path)):
        raise FileNotFoundError(f"Stats not found under: {stats_dir}")
    means = np.load(means_path)
    stds = np.load(stds_path)
    if means.ndim != 1 or stds.ndim != 1 or means.shape != stds.shape:
        raise RuntimeError("Invalid stats shapes; expected 1D arrays of same length")
    return means, stds


def read_timestamp_from_h5(h5_file: h5py.File, default_name: str) -> str:
    """Read timestamp from H5 file if present; otherwise fall back to filename stem."""
    if "timestamp" in h5_file:
        ds = h5_file["timestamp"]
        try:
            if ds.shape == ():
                val = ds[()]
            else:
                val = ds[:]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            if isinstance(val, np.ndarray) and val.dtype.type is np.bytes_:
                return val.tobytes().decode("utf-8")
            return str(val)
        except Exception:
            pass
    return os.path.splitext(default_name)[0]


def collect_targets(input_dir: str, date_str: str | None, datetime_str: str | None) -> List[str]:
    """Collect input .h5 file paths for a given date or a specific datetime string."""
    targets: List[str] = []
    if (date_str is None) == (datetime_str is None):
        raise ValueError("Specify exactly one of --date or --datetime for inference mode")

    if datetime_str:
        filename = f"{datetime_str}.h5"
        targets = [os.path.join(input_dir, filename)]
    else:
        for name in os.listdir(input_dir):
            if not (name.endswith(".h5") and name.startswith(date_str)):
                continue
            try:
                datetime.strptime(os.path.splitext(name)[0], "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            targets.append(os.path.join(input_dir, name))
        targets.sort()
    return targets


def build_sparsemae(in_chans: int, checkpoint_path: str, device: torch.device, repo_root: str, args_mask_ratio: float = 0.75) -> SparseMAE:
    """Instantiate pretrain model like experiment.py using pretrain params YAML, then load checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load pretrain params yaml
    cfg_path = os.path.join(repo_root, "ml", "params", "pretrain", "params.yaml")
    model_type = "vit_for_FT128db"
    common = {
        "in_chans": in_chans,
        "mask_ratio": args_mask_ratio,
        "stdwise": False,
        "pyramid": True,
        "sunspot": True,
        "base_mask_ratio": 0.5,
        "sunspot_spatial_ratio": 0.3,
        "feature_mask_ratio": 0.5,
    }
    try:
        with open(cfg_path, "r") as f:
            y = yaml.safe_load(f) or {}
        m = (y.get("model") or {})
        model_type = m.get("type", model_type)
        common["in_chans"] = int(m.get("in_chans", in_chans))
        common["mask_ratio"] = float(m.get("mask_ratio", args_mask_ratio))
        common["stdwise"] = bool(m.get("stdwise", False))
        common["pyramid"] = bool(m.get("pyramid", True))
        common["sunspot"] = bool(m.get("sunspot", True))
        common["base_mask_ratio"] = float(m.get("base_mask_ratio", 0.5))
        common["sunspot_spatial_ratio"] = float(m.get("sunspot_spatial_ratio", 0.3))
        common["feature_mask_ratio"] = float(m.get("feature_mask_ratio", 0.5))
    except Exception:
        pass

    # Instantiate same as experiment.py
    if model_type == "vit_for_FT128db":
        model = vit_for_FT128db(**common)
    elif model_type == "vit_for_FT64d4b":
        model = vit_for_FT64d4b(**common)
    elif model_type == "vit_for_FT32d4b":
        model = vit_for_FT32d4b(**common)
    elif model_type == "mae_vit_base_patch16_dec512d8b":
        model = mae_vit_base_patch16_dec512d8b(**common)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Load checkpoint non-strict
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state_dict = state[key]
                break
        else:
            state_dict = state
    else:
        state_dict = state

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:]] = v if not k.startswith("module.") else v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)} (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")

    model.eval()
    return model


def _ascend(path: str, levels: int) -> str:
    for _ in range(levels):
        path = os.path.dirname(path)
    return path


def run_pretrain_inference(args) -> None:
    """Run feature extraction using SparseMAE for the specified date/datetime."""
    # Resolve roots
    this_file = os.path.abspath(__file__)
    ml_dir = _ascend(this_file, 5)   # <repo>/ml
    repo_root = _ascend(this_file, 6)  # <repo>

    # Resolve data_root (relative paths are repo-root based)
    if os.path.isabs(args.data_root):
        data_root = args.data_root
    else:
        data_root = os.path.normpath(os.path.join(repo_root, args.data_root))

    input_h5_dir = os.path.join(data_root, "all_data_hours")
    output_dir = os.path.join(data_root, "all_features")

    # Stats for pretrain should come from data_root/pretrain/cache/fold{fold}/train
    stats_train_dir = os.path.join(data_root, "pretrain", "cache", f"fold{args.fold}", "train")

    os.makedirs(output_dir, exist_ok=True)

    means, stds = load_stats(stats_train_dir)
    in_chans = int(means.shape[0])

    device = (
        torch.device(f"cuda:{args.cuda_device}")
        if (getattr(args, "cuda_device", -1) >= 0 and torch.cuda.is_available())
        else torch.device("cpu")
    )

    ckpt_path = args.pretrain_checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(repo_root, ckpt_path))

    model = build_sparsemae(
        in_chans=in_chans,
        checkpoint_path=ckpt_path,
        device=device,
        repo_root=repo_root,
        args_mask_ratio=0.75,
    )

    targets = collect_targets(input_h5_dir, getattr(args, "date", None), getattr(args, "datetime", None))
    if not targets:
        print(f"Info: no input files found under {input_h5_dir}")
        return

    for h5_path in targets:
        base = os.path.basename(h5_path)
        out_path = os.path.join(output_dir, base)
        if os.path.exists(out_path):
            print(f"Skip existing: {out_path}")
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                X = f["X"][:]
                ts = read_timestamp_from_h5(f, base)
        except Exception as e:
            print(f"Error reading {h5_path}: {e}")
            continue

        if X.ndim != 3 or X.shape[0] != in_chans:
            print(f"Error: unexpected X shape {X.shape} in {base}; expected ({in_chans}, H, W)")
            continue

        Xn = (X - means[:, None, None]) / (stds[:, None, None] + 1e-8)
        tensor = torch.from_numpy(Xn).float().unsqueeze(0).to(device)

        with torch.no_grad():
            try:
                tokens, _, _ = model.forward_encoder_pyramid(tensor, mask_ratio=0.0)
                feat = tokens[:, 1:, :].mean(dim=1).squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"Error during model forward for {base}: {e}")
                continue

        try:
            with h5py.File(out_path, "w") as f:
                f.create_dataset("features", data=feat)
                f.create_dataset("timestamp", data=str(ts).encode("utf-8"))
                f.attrs["original_shape"] = X.shape
                f.attrs["features_shape"] = feat.shape
            print(f"Saved features: {out_path}")
        except Exception as e:
            print(f"Error writing {out_path}: {e}") 
