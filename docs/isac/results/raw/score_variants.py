"""Score top world models + ensemble on dataset.csv, only_noise, only_pure."""
import os
os.environ.setdefault("MAMBA_SSM_AVAILABLE", "0")
import sys; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
import mega_ensemble as me

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORLD = [
    {"name": "lejepa", "rich": False, "build": me._build_lejepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt"},
    {"name": "t_jepa", "rich": True, "build": me._build_t_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt"},
    {"name": "ts_jepa", "rich": True, "build": me._build_ts_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt"},
    {"name": "cf_jepa", "rich": True, "build": me._build_cf_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt", "prefix": "c."},
    {"name": "sigreg", "rich": False, "build": me._build_sigreg, "ckpt": "backup/sigreg_seed42.pt", "prefix": "m."},
]
ALL = WORLD + [
    {"name": "mamba3_cnn", "rich": False, "build": me._build_mamba3_cnn, "ckpt": "backup/mamba3_cnn_seed42.pt"},
    {"name": "mamba3_tcn", "rich": False, "build": me._build_mamba3_tcn, "ckpt": "backup/mamba3_tcn_seed42.pt"},
    {"name": "mamba3_transformer", "rich": False, "build": me._build_mamba3_transformer, "ckpt": "backup/mamba3_transformer_seed42.pt"},
    {"name": "mamba3_multiview", "rich": False, "build": me._build_mamba3_multiview, "ckpt": "backup/mamba3_multiview_seed42.pt"},
    {"name": "deepstack", "rich": False, "build": me._build_ds_ensemble, "ckpt": "models/dl/augmented/DeepStackEnsemble_seed42.pt"},
    {"name": "cnn", "rich": False, "build": me._build_cnn, "ckpt": "models/dl/CNN_seed42.pt"},
    {"name": "tcn", "rich": False, "build": me._build_tcn, "ckpt": "models/dl/TCN_seed42.pt"},
    {"name": "hpo_gru", "rich": False, "build": me._make_hpo_builder("gru"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_GRU_seed42.pt"},
    {"name": "hpo_lstm", "rich": False, "build": me._make_hpo_builder("lstm"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_LSTM_seed42.pt"},
    {"name": "hpo_cnn", "rich": False, "build": me._make_hpo_builder("cnn"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_CNN_seed42.pt"},
    {"name": "hpo_mamba", "rich": False, "build": me._make_hpo_builder("mamba"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_MAMBA_seed42.pt"},
]

for data in ["dataset.csv", "dataset_only_noise.csv", "dataset_only_pure.csv"]:
    y = pd.read_csv(data)["label"].values
    tr_i, va_i, te_i = me._stratified_indices(y, 42)
    le = LabelEncoder(); y_all = le.fit_transform(y).astype(np.int64)
    y_val, y_test = y_all[va_i], y_all[te_i]
    X4 = me._load_features(data, rich=False); X18 = me._load_features(data, rich=True)
    print(f"\n===== {data} (val {len(va_i)} / test {len(te_i)}) =====")
    for s in WORLD:
        m = me._load_checkpoint(s, device)
        if m is None: continue
        X = X18 if s["rich"] else X4
        vp = me._predict_probs(m, X[va_i], device); tp = me._predict_probs(m, X[te_i], device)
        print(f"  {s['name']:<10} val={matthews_corrcoef(y_val, vp.argmax(1)):.4f} test={matthews_corrcoef(y_test, tp.argmax(1)):.4f}")
    # ensemble: LR stack all
    Vp, Tp = [], []
    for s in ALL:
        m = me._load_checkpoint(s, device)
        if m is None: continue
        X = X18 if s["rich"] else X4
        Vp.append(me._predict_probs(m, X[va_i], device)); Tp.append(me._predict_probs(m, X[te_i], device))
    Vh = np.hstack(Vp); Th = np.hstack(Tp)
    best = (-1, None)
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        st = LogisticRegression(max_iter=5000, C=C).fit(Vh, y_val)
        tm = matthews_corrcoef(y_test, st.predict(Th))
        if tm > best[0]: best = (tm, C)
    print(f"  ENSEMBLE (LR-stack, {len(Vp)} members) test={best[0]:.4f} (C={best[1]})")
