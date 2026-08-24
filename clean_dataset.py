#!/usr/bin/env python3
"""Dataset cleaning audit — find and drop devices/samples that poison training.

Why: on the current dataset.csv the absolute RSSI level carries ~3x more
information about WHICH DEVICE recorded a sequence than about WHICH movement it
is (MI with MAC 0.219 vs MI with label 0.071). Several devices sit at/below
chance under cross-device evaluation, i.e. their sequences genuinely do not
look like their labels. This script quantifies that per device, drops the
poisoned parts, and writes a cleaned CSV plus a full report.

Per-device diagnostics
----------------------
within_mcc : stratified CV *inside* the device (does it agree with itself?)
             low  -> labels are inconsistent/noisy for this device  -> DROP
lodo_mcc   : leave-one-device-out (train on all other devices, predict this one)
             low  -> device distribution is shifted vs everyone else -> DROP
             (unless --keep-shifted: keep self-consistent-but-shifted devices)

Sample-level diagnostic
-----------------------
OOF probabilities from a KNN on the surviving rows: samples the model rejects
with high confidence (max prob > --ambiguity-thresh AND wrong class) are
reported to <out-stem>_flagged.csv; dropped too with --drop-ambiguous.

Usage
-----
    MAMBA_SSM_AVAILABLE=0 uv run python clean_dataset.py --data dataset.csv
    # stricter: also drop ambiguous samples
    MAMBA_SSM_AVAILABLE=0 uv run python clean_dataset.py --drop-ambiguous
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FEATS = [str(i) for i in range(1, 11)]


def _zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _cv_preds(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
              proba: bool = False) -> np.ndarray:
    n_min = pd.Series(y).value_counts().min()
    k = max(2, min(n_splits, int(n_min)))
    skf = StratifiedKFold(k, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=skf, method="predict_proba" if proba else "predict")


def _mi_binned(x: np.ndarray, codes: np.ndarray, bins: int = 20) -> float:
    b = pd.cut(x, bins=bins, labels=False)
    return float(mutual_info_score(b, codes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset.csv"))
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: <stem>_clean.csv)")
    ap.add_argument("--report-dir", type=Path, default=Path("results"))
    ap.add_argument("--within-floor", type=float, default=0.15,
                    help="drop device if within-device CV MCC < this")
    ap.add_argument("--lodo-floor", type=float, default=0.05,
                    help="drop device if leave-one-device-out MCC < this")
    ap.add_argument("--keep-shifted", action="store_true",
                    help="keep self-consistent devices even if LODO says shifted")
    ap.add_argument("--drop-ambiguous", action="store_true",
                    help="also drop samples flagged by OOF disagreement")
    ap.add_argument("--ambiguity-thresh", type=float, default=0.85)
    args = ap.parse_args()

    out_path = args.out or args.data.with_name(args.data.stem + "_clean.csv")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    y = df["label"].values
    macs = df["mac"].values
    raw = df[FEATS].values.astype(np.float32)
    X = _zscore(raw)
    mac_codes = pd.Categorical(macs).codes
    lab_codes = pd.Categorical(y).codes

    # ── Baselines BEFORE cleaning ────────────────────────────────────────────
    knn = KNeighborsClassifier(5)
    knn_oof = _cv_preds(knn, X, y)
    base_knn = matthews_corrcoef(y, knn_oof)
    log_oof = _cv_preds(LogisticRegression(max_iter=3000, C=0.5), X, y)
    base_log = matthews_corrcoef(y, log_oof)
    mi_mac = _mi_binned(raw.mean(1), mac_codes)
    mi_lab = _mi_binned(raw.mean(1), lab_codes)

    print("=" * 78)
    print(f"Dataset: {args.data}  ({len(df)} rows, {len(set(macs))} devices)")
    print(f"BEFORE:  KNN-5 CV MCC={base_knn:.3f}   LogReg CV MCC={base_log:.3f}")
    print(f"Confounding: MI(mean-RSSI, MAC)={mi_mac:.3f}  vs  MI(mean-RSSI, label)={mi_lab:.3f}")
    print("=" * 78)

    # ── Per-device audit ─────────────────────────────────────────────────────
    rows = []
    for mac in sorted(set(macs)):
        m = macs == mac
        Xd, yd, rd = X[m], y[m], raw[m]
        counts = pd.Series(yd).value_counts()

        # within-device self-consistency
        if len(counts) > 1 and counts.min() >= 2:
            pred_w = _cv_preds(KNeighborsClassifier(3), Xd, yd)
            within = matthews_corrcoef(yd, pred_w)
        else:
            within = float("nan")  # degenerate (single-class or too few)

        # leave-one-device-out
        clf = KNeighborsClassifier(5).fit(X[~m], y[~m])
        pl = clf.predict(Xd)
        lodo = matthews_corrcoef(yd, pl)
        lodo_acc = float((pl == yd).mean())

        inconsistent = (not np.isnan(within)) and within < args.within_floor
        shifted = lodo < args.lodo_floor
        drop = inconsistent or (shifted and not args.keep_shifted)

        rows.append({
            "mac": mac, "n": int(m.sum()), "n_classes": len(counts),
            "rssi_mean": round(float(rd.mean()), 1),
            "within_mcc": None if np.isnan(within) else round(within, 3),
            "lodo_mcc": round(lodo, 3), "lodo_acc": round(lodo_acc, 3),
            "label_inconsistent": bool(inconsistent),
            "dist_shifted": bool(shifted),
            "decision": "DROP" if drop else "keep",
        })
    audit = pd.DataFrame(rows)
    with pd.option_context("display.width", 160):
        print("\n=== Per-device audit ===")
        print(audit.to_string(index=False))

    drop_set = set(audit.loc[audit.decision == "DROP", "mac"])
    survivors = df[~df["mac"].isin(drop_set)].reset_index(drop=True)
    n_drop_dev = len(drop_set)

    # ── Sample-level ambiguity flagging on survivors ────────────────────────
    flagged_path = out_path.with_name(out_path.stem + "_flagged.csv")
    n_flagged = 0
    if len(survivors) > 0 and survivors["label"].nunique() > 1:
        Xs = _zscore(survivors[FEATS].values.astype(np.float32))
        ys = survivors["label"].values
        proba = _cv_preds(KNeighborsClassifier(5), Xs, ys, proba=True)
        pred = proba.argmax(1)
        conf = proba.max(1)
        amb = (conf > args.ambiguity_thresh) & (pred != ys)
        flagged = survivors.loc[amb].copy()
        class_names = np.array(sorted(set(ys)))
        flagged["oof_pred"] = class_names[pred[amb]]
        flagged["oof_conf"] = conf[amb].round(3)
        flagged.to_csv(flagged_path, index=False)
        n_flagged = int(amb.sum())
        print(f"\nAmbiguous samples (OOF conf>{args.ambiguity_thresh} AND wrong): "
              f"{n_flagged}  → {flagged_path}")
        if args.drop_ambiguous and n_flagged:
            survivors = survivors.loc[~amb].reset_index(drop=True)
            print(f"Dropped {n_flagged} ambiguous samples (--drop-ambiguous)")

    # ── AFTER cleaning ───────────────────────────────────────────────────────
    ys2 = survivors["label"].values
    Xs2 = _zscore(survivors[FEATS].values.astype(np.float32))
    mac2 = pd.Categorical(survivors["mac"]).codes
    knn_after = matthews_corrcoef(ys2, _cv_preds(KNeighborsClassifier(5), Xs2, ys2)) \
        if survivors["label"].nunique() > 1 else float("nan")
    log_after = matthews_corrcoef(ys2, _cv_preds(LogisticRegression(max_iter=3000, C=0.5), Xs2, ys2)) \
        if survivors["label"].nunique() > 1 else float("nan")
    mi_mac_a = _mi_binned(survivors[FEATS].values.astype(np.float32).mean(1), mac2)

    print("\n=== Result ===")
    print(f"Devices: {len(set(macs))} -> {len(set(survivors['mac']))}  "
          f"(dropped {n_drop_dev}: {', '.join(sorted(drop_set)) or '-'})")
    print(f"Rows:    {len(df)} -> {len(survivors)}")
    print(f"KNN-5 CV MCC:   {base_knn:.3f} -> {knn_after:.3f}")
    print(f"LogReg CV MCC:  {base_log:.3f} -> {log_after:.3f}")
    print(f"MI(rssi,MAC):   {mi_mac:.3f} -> {mi_mac_a:.3f}   "
          f"(MI(rssi,label) was {mi_lab:.3f})")

    survivors.to_csv(out_path, index=False)
    audit.to_csv(args.report_dir / "device_audit.csv", index=False)
    print(f"\nWrote {out_path}, {args.report_dir/'device_audit.csv'}")

    # ── Recommendation ───────────────────────────────────────────────────────
    gain = knn_after - base_knn
    if gain > 0.10:
        print(f"\nVerdict: cleaning helped a lot (+{gain:.2f} KNN MCC). "
              f"Retrain/HPO against {out_path}.")
    elif gain > 0.02:
        print(f"\nVerdict: cleaning helped somewhat (+{gain:.2f}). "
              f"Worth retraining, but also inspect flagged samples for relabeling.")
    else:
        print("\nVerdict: cleaning barely moved the ceiling. The remaining confusion "
              "is likely genuine class overlap (AA-vs-BA style), not bad devices — "
              "consider relabeling flagged samples instead.")


if __name__ == "__main__":
    main()
