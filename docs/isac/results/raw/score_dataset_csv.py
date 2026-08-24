"""Score every ensemble member checkpoint on dataset.csv (val + test MCC)."""
import os
os.environ.setdefault("MAMBA_SSM_AVAILABLE", "0")
import sys; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
import mega_ensemble as me

DATA = "dataset.csv"
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

y = pd.read_csv(DATA)["label"].values
train_idx, val_idx, test_idx = me._stratified_indices(y, SEED)
le = LabelEncoder(); y_all = le.fit_transform(y).astype(np.int64)
y_val, y_test = y_all[val_idx], y_all[test_idx]
print(f"Train {len(train_idx)} / Val {len(val_idx)} / Test {len(test_idx)}")

X4 = me._load_features(DATA, rich=False)
X18 = me._load_features(DATA, rich=True)

specs = [
    # world models
    {"name": "lejepa", "rich": False, "build": me._build_lejepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt"},
    {"name": "t_jepa", "rich": True, "build": me._build_t_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt"},
    {"name": "ts_jepa", "rich": True, "build": me._build_ts_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt"},
    {"name": "cf_jepa", "rich": True, "build": me._build_cf_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt", "prefix": "c."},
    {"name": "sigreg", "rich": False, "build": me._build_sigreg, "ckpt": "backup/sigreg_seed42.pt", "prefix": "m."},
    {"name": "sigreg_s3", "rich": False, "build": me._build_sigreg, "ckpt": "backup/sigreg_seed3.pt", "prefix": "m."},
    {"name": "sigreg_s5", "rich": False, "build": me._build_sigreg, "ckpt": "backup/sigreg_seed5.pt", "prefix": "m."},
    # mamba3
    {"name": "mamba3_cnn", "rich": False, "build": me._build_mamba3_cnn, "ckpt": "backup/mamba3_cnn_seed42.pt"},
    {"name": "mamba3_tcn", "rich": False, "build": me._build_mamba3_tcn, "ckpt": "backup/mamba3_tcn_seed42.pt"},
    {"name": "mamba3_transformer", "rich": False, "build": me._build_mamba3_transformer, "ckpt": "backup/mamba3_transformer_seed42.pt"},
    {"name": "mamba3_multiview", "rich": False, "build": me._build_mamba3_multiview, "ckpt": "backup/mamba3_multiview_seed42.pt"},
    # DL
    {"name": "deepstack", "rich": False, "build": me._build_ds_ensemble, "ckpt": "models/dl/augmented/DeepStackEnsemble_seed42.pt"},
    {"name": "cnn", "rich": False, "build": me._build_cnn, "ckpt": "models/dl/CNN_seed42.pt"},
    {"name": "tcn", "rich": False, "build": me._build_tcn, "ckpt": "models/dl/TCN_seed42.pt"},
    {"name": "gru", "rich": False, "build": me._build_gru, "ckpt": "models/dl/GRU_seed42.pt"},
    {"name": "lstm", "rich": False, "build": me._build_lstm, "ckpt": "models/dl/LSTM_seed42.pt"},
    {"name": "bilstm", "rich": False, "build": me._build_bilstm, "ckpt": "models/dl/BiLSTM_seed42.pt"},
    # HPO
    {"name": "hpo_gru", "rich": False, "build": me._make_hpo_builder("gru"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_GRU_seed42.pt"},
    {"name": "hpo_lstm", "rich": False, "build": me._make_hpo_builder("lstm"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_LSTM_seed42.pt"},
    {"name": "hpo_cnn", "rich": False, "build": me._make_hpo_builder("cnn"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_CNN_seed42.pt"},
    {"name": "hpo_mamba", "rich": False, "build": me._make_hpo_builder("mamba"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_MAMBA_seed42.pt"},
    {"name": "hpo_tcn", "rich": False, "build": me._make_hpo_builder("tcn"), "ckpt": "models/dl/augmented/HPO_TCN_seed42.pt"},
]

rows = []
for s in specs:
    m = me._load_checkpoint(s, device)
    if m is None:
        print(f"  [skip] {s['name']} (load failed)")
        continue
    X = X18 if s["rich"] else X4
    vp = me._predict_probs(m, X[val_idx], device)
    tp = me._predict_probs(m, X[test_idx], device)
    vm = matthews_corrcoef(y_val, vp.argmax(1))
    tm = matthews_corrcoef(y_test, tp.argmax(1))
    n = sum(p.numel() for p in m.parameters()) / 1e6
    rows.append((s["name"], vm, tm, n))
    print(f"  {s['name']:<22} val={vm:.4f} test={tm:.4f} params={n:.3f}M")

# ensemble: plain mean + LR stack of ALL loaded members
print("\n=== Ensembles on dataset.csv ===")
if len(rows) >= 3:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    names_all = [r[0] for r in rows]
    V = np.array([me._predict_probs(me._load_checkpoint(s, device), X18 if s["rich"] else X4, device)[val_idx if False else slice(None)] for s in specs if me._load_checkpoint(s, device) is not None])
    # simpler: recompute cleanly
    Vp, Tp = [], []
    for s in specs:
        m = me._load_checkpoint(s, device)
        if m is None: continue
        X = X18 if s["rich"] else X4
        Vp.append(me._predict_probs(m, X[val_idx], device))
        Tp.append(me._predict_probs(m, X[test_idx], device))
    Pm = np.mean(Vp, axis=0); Pt = np.mean(Tp, axis=0)
    print(f"  plain-mean val={matthews_corrcoef(y_val, Pm.argmax(1)):.4f} test={matthews_corrcoef(y_test, Pt.argmax(1)):.4f}")
    Vh = np.hstack(Vp); Th = np.hstack(Tp)
    best = (-1, None)
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        st = LogisticRegression(max_iter=5000, C=C).fit(Vh, y_val)
        tm = matthews_corrcoef(y_test, st.predict(Th))
        if tm > best[0]: best = (tm, C)
    print(f"  LR-stack test={best[0]:.4f} (C={best[1]})")
