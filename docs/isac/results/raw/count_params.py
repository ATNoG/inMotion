"""Count params for every model in the master results file."""
import os
os.environ.setdefault("MAMBA_SSM_AVAILABLE", "0")
import sys; sys.path.insert(0, os.getcwd())
import torch
import mega_ensemble as me

dev = torch.device("cpu")

specs = [
    # world models (jepa + sigreg)
    {"name": "lejepa (full-after-hpo)", "build": me._build_lejepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt"},
    {"name": "t_jepa (full-after-hpo)", "build": me._build_t_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt"},
    {"name": "ts_jepa (base)", "build": me._build_ts_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt"},
    {"name": "cf_jepa (full-after-hpo)", "build": me._build_cf_jepa, "ckpt": "20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt", "prefix": "c."},
    {"name": "sigreg (backup s42)", "build": me._build_sigreg, "ckpt": "backup/sigreg_seed42.pt", "prefix": "m."},
    {"name": "sigreg_s3", "build": me._build_sigreg, "ckpt": "backup/sigreg_seed3.pt", "prefix": "m."},
    {"name": "sigreg_s5", "build": me._build_sigreg, "ckpt": "backup/sigreg_seed5.pt", "prefix": "m."},
    # mamba3 family
    {"name": "mamba3_cnn", "build": me._build_mamba3_cnn, "ckpt": "backup/mamba3_cnn_seed42.pt"},
    {"name": "mamba3_tcn", "build": me._build_mamba3_tcn, "ckpt": "backup/mamba3_tcn_seed42.pt"},
    {"name": "mamba3_transformer", "build": me._build_mamba3_transformer, "ckpt": "backup/mamba3_transformer_seed42.pt"},
    {"name": "mamba3_multiview", "build": me._build_mamba3_multiview, "ckpt": "backup/mamba3_multiview_seed42.pt"},
    # DL
    {"name": "deepstack", "build": me._build_ds_ensemble, "ckpt": "models/dl/augmented/DeepStackEnsemble_seed42.pt"},
    {"name": "cnn", "build": me._build_cnn, "ckpt": "models/dl/CNN_seed42.pt"},
    {"name": "tcn", "build": me._build_tcn, "ckpt": "models/dl/TCN_seed42.pt"},
    {"name": "gru", "build": me._build_gru, "ckpt": "models/dl/GRU_seed42.pt"},
    {"name": "lstm", "build": me._build_lstm, "ckpt": "models/dl/LSTM_seed42.pt"},
    {"name": "bilstm", "build": me._build_bilstm, "ckpt": "models/dl/BiLSTM_seed42.pt"},
    # HPO models
    {"name": "hpo_gru", "build": me._make_hpo_builder("gru"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_GRU_seed42.pt"},
    {"name": "hpo_lstm", "build": me._make_hpo_builder("lstm"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_LSTM_seed42.pt"},
    {"name": "hpo_cnn", "build": me._make_hpo_builder("cnn"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_CNN_seed42.pt"},
    {"name": "hpo_mamba", "build": me._make_hpo_builder("mamba"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_MAMBA_seed42.pt"},
    {"name": "hpo_tcn", "build": me._make_hpo_builder("tcn"), "ckpt": "20-aug/models/dl/normal-new-ds/HPO_TCN_seed42.pt"},
]

rows = []
for s in specs:
    m = me._load_checkpoint(s, dev)
    if m is None:
        # try building without ckpt load
        try:
            m = s["build"]()
            n = sum(p.numel() for p in m.parameters())
            rows.append((s["name"], n / 1e6, "BUILD-ONLY"))
        except Exception as e:
            rows.append((s["name"], None, f"FAIL {str(e)[:60]}"))
    else:
        n = sum(p.numel() for p in m.parameters())
        rows.append((s["name"], n / 1e6, "CKPT"))

for name, n, src in rows:
    if n is None:
        print(f"  {name:<30} {src}")
    else:
        print(f"  {name:<30} {n:8.3f} M  ({src})")
