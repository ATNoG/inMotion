"""Permutation importance of the 18 rich feature channels.

For each rich-feature model (t_jepa, ts_jepa, cf_jepa), compute the val/test
MCC drop when each of the 18 channels is shuffled independently. Bigger drop
= more important channel. Uses the standard 80/20 split (seed 42) on
dataset.icaisf.csv.
"""
import os
os.environ.setdefault("MAMBA_SSM_AVAILABLE", "0")
import sys; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
import mega_ensemble as me

DATA = "dataset.icaisf.csv"
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)

FEATURE_NAMES = [
    "raw", "diff1", "diff2", "dev",
    "band_dc", "band_ac",
    "roll_mean", "roll_range", "roll_std", "roll_skew", "roll_kurt",
    "peak_loc", "slope_sign", "sign_change",
    "stft_b0", "stft_b1", "stft_b2", "stft_b3",
]

y = pd.read_csv(DATA)["label"].values
tr_i, va_i, te_i = me._stratified_indices(y, SEED)
le = LabelEncoder(); y_all = le.fit_transform(y).astype(np.int64)
y_val, y_test = y_all[va_i], y_all[te_i]
X18 = me._load_features(DATA, rich=True)
print(f"X18 shape: {X18.shape} (N, T=10, C=18)")

MODELS = [
    ("t_jepa", me._build_t_jepa, "20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt"),
    ("ts_jepa", me._build_ts_jepa, "20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt"),
    ("cf_jepa", me._build_cf_jepa, "20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt"),
]

for name, build, ckpt in MODELS:
    spec = {"name": name, "rich": True, "build": build, "ckpt": ckpt}
    if name == "cf_jepa":
        spec["prefix"] = "c."
    m = me._load_checkpoint(spec, device)
    if m is None:
        print(f"[skip] {name} failed to load")
        continue
    base_v = matthews_corrcoef(y_val, me._predict_probs(m, X18[va_i], device).argmax(1))
    base_t = matthews_corrcoef(y_test, me._predict_probs(m, X18[te_i], device).argmax(1))
    print(f"\n=== {name} (baseline val={base_v:.4f} test={base_t:.4f}) ===")
    print(f"  {'ch':<12} {'val_drop':>8} {'test_drop':>9}  {'rank'}")
    drops = []
    for c in range(18):
        Xp_val = X18[va_i].copy(); Xp_test = X18[te_i].copy()
        perm = rng.permutation(len(Xp_val))
        Xp_val[:, :, c] = Xp_val[perm][:, :, c]
        perm = rng.permutation(len(Xp_test))
        Xp_test[:, :, c] = Xp_test[perm][:, :, c]
        vp = me._predict_probs(m, Xp_val, device)
        tp = me._predict_probs(m, Xp_test, device)
        vm = matthews_corrcoef(y_val, vp.argmax(1))
        tm = matthews_corrcoef(y_test, tp.argmax(1))
        drops.append((base_v - vm, base_t - tm, c))
    for rank, (vd, td, c) in enumerate(sorted(drops, key=lambda x: -x[1]), 1):
        print(f"  {FEATURE_NAMES[c]:<12} {vd:8.4f} {td:9.4f}  {rank}")
