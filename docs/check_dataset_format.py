"""Validate a feature-contract dataset folder before handing it to dataset.py / train.py.
See FEATURE_CONTRACT.md for the format this checks against.

Usage:
    python check_dataset_format.py /path/to/data/mafw
    python check_dataset_format.py /path/to/data/mafw --full   # check every file, not a sample
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch


def fail(msg: str) -> bool:
    print(f"[FAIL] {msg}")
    return False


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def check(root: Path, full: bool) -> bool:
    ok = True
    meta_path, labels_path = root / "meta.json", root / "labels.csv"
    audio_dir, visual_dir = root / "audio", root / "visual"

    for p, name in [(meta_path, "meta.json"), (labels_path, "labels.csv"),
                    (audio_dir, "audio/"), (visual_dir, "visual/")]:
        if not p.exists():
            ok = fail(f"Missing {name} at {p}") and ok
    if not ok:
        return False

    meta = json.loads(meta_path.read_text())
    for key in ["dataset_name", "num_classes", "class_names", "audio_dim", "visual_dim"]:
        if key not in meta:
            ok = fail(f"meta.json missing key '{key}'") and ok
    if not ok:
        return False
    if len(meta["class_names"]) != meta["num_classes"]:
        ok = fail(f"class_names has {len(meta['class_names'])} entries but "
                  f"num_classes={meta['num_classes']}") and ok

    df = pd.read_csv(labels_path)
    for col in ["clip_id", "label", "fold"]:
        if col not in df.columns:
            ok = fail(f"labels.csv missing column '{col}'") and ok
    if not ok:
        return False

    if df["clip_id"].duplicated().any():
        dupes = df.loc[df["clip_id"].duplicated(), "clip_id"].tolist()[:5]
        ok = fail(f"Duplicate clip_id in labels.csv, e.g. {dupes}") and ok

    n_classes = meta["num_classes"]
    bad_labels = df[(df["label"] < 0) | (df["label"] >= n_classes)]
    if len(bad_labels):
        ok = fail(f"{len(bad_labels)} rows have label outside [0,{n_classes - 1}]") and ok

    bad_folds = df[(df["fold"] < 1) | (df["fold"] > 5)]
    if len(bad_folds):
        ok = fail(f"{len(bad_folds)} rows have fold outside [1,5]") and ok

    print("Class distribution (label -> count):")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        name = meta["class_names"][lbl] if 0 <= lbl < len(meta["class_names"]) else "?"
        print(f"  {lbl} ({name}): {cnt}")
    print("Fold distribution:", dict(df["fold"].value_counts().sort_index()))

    audio_dim, visual_dim = meta["audio_dim"], meta["visual_dim"]
    all_ids = df["clip_id"].astype(str).tolist()
    ids_to_check = all_ids
    if not full and len(all_ids) > 200:
        warn(f"{len(all_ids)} clips total, spot-checking a random 200 for speed "
             f"(pass --full to check every file)")
        random.seed(0)
        ids_to_check = random.sample(all_ids, 200)

    n_checked = 0
    missing_audio, missing_visual, bad_shape, bad_dtype, bad_values = [], [], [], [], []

    for cid in ids_to_check:
        a_path, v_path = audio_dir / f"{cid}.pt", visual_dir / f"{cid}.pt"
        if not a_path.exists():
            missing_audio.append(cid)
            continue
        if not v_path.exists():
            missing_visual.append(cid)
            continue
        a = torch.load(a_path, map_location="cpu")
        v = torch.load(v_path, map_location="cpu")
        n_checked += 1
        if a.dim() != 2 or a.shape[1] != audio_dim:
            bad_shape.append((cid, "audio", tuple(a.shape)))
        if v.dim() != 2 or v.shape[1] != visual_dim:
            bad_shape.append((cid, "visual", tuple(v.shape)))
        if a.dtype != torch.float32 or v.dtype != torch.float32:
            bad_dtype.append(cid)
        if torch.isnan(a).any() or torch.isinf(a).any() or torch.isnan(v).any() or torch.isinf(v).any():
            bad_values.append(cid)

    if missing_audio:
        ok = fail(f"{len(missing_audio)} clip_id have no matching audio/*.pt, e.g. {missing_audio[:5]}") and ok
    if missing_visual:
        ok = fail(f"{len(missing_visual)} clip_id have no matching visual/*.pt, e.g. {missing_visual[:5]}") and ok
    if bad_shape:
        ok = fail(f"{len(bad_shape)} tensors with wrong ndim/feature-dim, e.g. {bad_shape[:5]}") and ok
    if bad_dtype:
        ok = fail(f"{len(bad_dtype)} tensors not float32, e.g. {bad_dtype[:5]}") and ok
    if bad_values:
        ok = fail(f"{len(bad_values)} tensors contain NaN/Inf, e.g. {bad_values[:5]}") and ok

    print(f"\nChecked {n_checked}/{len(ids_to_check)} sampled clips (dataset has {len(all_ids)} total).")
    print("RESULT:", "PASS - an toan de train" if ok else "FAIL - sua loi tren truoc khi train")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=str)
    p.add_argument("--full", action="store_true", help="check every clip instead of a 200-sample")
    args = p.parse_args()
    sys.exit(0 if check(Path(args.dataset_root), args.full) else 1)
