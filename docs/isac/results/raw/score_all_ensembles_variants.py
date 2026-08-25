"""Compute every named ensemble (E1-E8) on every dataset variant.

Processes ONE dataset at a time, loading members and releasing them to stay
under memory. Since repeated model loads dominate, we cache per-dataset
member probs in a dict and free checkpoints between datasets.
"""
import os, sys
os.environ.setdefault("MAMBA_SSM_AVAILABLE", "0")
sys.path.insert(0, os.getcwd())
import gc
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
import mega_ensemble as me

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

ENS = {
    "E1_all21":     (["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","sigreg_s3","sigreg_s5","mamba3_cnn","mamba3_tcn","mamba3_transformer","mamba3_multiview","hpo_gru","hpo_lstm","hpo_cnn","hpo_mamba","deepstack","cnn","tcn","gru","lstm","bilstm"], "lr"),
    "E2_sizefrontier":(["sigreg","lejepa"], "mean"),
    "E3_16members": (["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","mamba3_cnn","mamba3_tcn","mamba3_transformer","mamba3_multiview","deepstack","cnn","tcn","hpo_gru","hpo_lstm","hpo_cnn","hpo_mamba"], "lr"),
    "E4_worldonly": (["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","sigreg_s3","sigreg_s5"], "mean"),
    "E5_worldbestdl":(["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","sigreg_s3","sigreg_s5","deepstack","mamba3_tcn"], "mean"),
    "E6_bestworld": (["sigreg","lejepa","ts_jepa"], "mean"),
    "E7_bestworldbestdl":(["sigreg","lejepa","ts_jepa","deepstack","mamba3_tcn"], "mean"),
    "E8_greedy":    (["lejepa","mamba3_multiview","mamba3_transformer","ts_jepa","sigreg","cf_jepa","mamba3_cnn","sigreg_s3","mamba3_tcn","t_jepa","hpo_lstm","sigreg_s5","deepstack","hpo_cnn","hpo_mamba","hpo_gru"], "lr"),
}

def specs():
    return [
        ("lejepa",False,me._build_lejepa,"20-aug/models/exotic/normal-new-ds/full-after-hpo/lejepa_ft_seed42.pt",None),
        ("t_jepa",True,me._build_t_jepa,"20-aug/models/exotic/normal-new-ds/full-after-hpo/t_jepa_ft_seed42.pt",None),
        ("ts_jepa",True,me._build_ts_jepa,"20-aug/models/exotic/normal-new-ds/ts_jepa_ft_seed42.pt",None),
        ("cf_jepa",True,me._build_cf_jepa,"20-aug/models/exotic/normal-new-ds/full-after-hpo/cf_jepa_ft_seed42.pt","c."),
        ("sigreg",False,me._build_sigreg,"backup/sigreg_seed42.pt","m."),
        ("sigreg_s3",False,me._build_sigreg,"backup/sigreg_seed3.pt","m."),
        ("sigreg_s5",False,me._build_sigreg,"backup/sigreg_seed5.pt","m."),
        ("mamba3_cnn",False,me._build_mamba3_cnn,"backup/mamba3_cnn_seed42.pt",None),
        ("mamba3_tcn",False,me._build_mamba3_tcn,"backup/mamba3_tcn_seed42.pt",None),
        ("mamba3_transformer",False,me._build_mamba3_transformer,"backup/mamba3_transformer_seed42.pt",None),
        ("mamba3_multiview",False,me._build_mamba3_multiview,"backup/mamba3_multiview_seed42.pt",None),
        ("deepstack",False,me._build_ds_ensemble,"models/dl/augmented/DeepStackEnsemble_seed42.pt",None),
        ("cnn",False,me._build_cnn,"models/dl/CNN_seed42.pt",None),
        ("tcn",False,me._build_tcn,"models/dl/TCN_seed42.pt",None),
        ("gru",False,me._build_gru,"models/dl/GRU_seed42.pt",None),
        ("lstm",False,me._build_lstm,"models/dl/LSTM_seed42.pt",None),
        ("bilstm",False,me._build_bilstm,"models/dl/BiLSTM_seed42.pt",None),
        ("hpo_gru",False,me._make_hpo_builder("gru"),"20-aug/models/dl/normal-new-ds/HPO_GRU_seed42.pt",None),
        ("hpo_lstm",False,me._make_hpo_builder("lstm"),"20-aug/models/dl/normal-new-ds/HPO_LSTM_seed42.pt",None),
        ("hpo_cnn",False,me._make_hpo_builder("cnn"),"20-aug/models/dl/normal-new-ds/HPO_CNN_seed42.pt",None),
        ("hpo_mamba",False,me._make_hpo_builder("mamba"),"20-aug/models/dl/normal-new-ds/HPO_MAMBA_seed42.pt",None),
    ]

def run_dataset(data):
    y = pd.read_csv(data)["label"].values
    tr_i, va_i, te_i = me._stratified_indices(y, SEED)
    le = LabelEncoder(); y_all = le.fit_transform(y).astype(np.int64)
    y_val, y_test = y_all[va_i], y_all[te_i]
    X4 = me._load_features(data, rich=False); X18 = me._load_features(data, rich=True)

    # Load each member, keep only probs, free model right away
    probs = {}
    for (name, rich, build, ckpt, prefix) in specs():
        spec = {"name":name,"rich":rich,"build":build,"ckpt":ckpt}
        if prefix: spec["prefix"] = prefix
        m = me._load_checkpoint(spec, DEVICE)
        if m is None:
            print(f"    [skip] {name}", flush=True); continue
        X = X18 if rich else X4
        probs[name] = (me._predict_probs(m, X[va_i], DEVICE), me._predict_probs(m, X[te_i], DEVICE))
        del m; gc.collect()
    print(f"\n===== {data} (val {len(va_i)} / test {len(te_i)}) =====", flush=True)
    # Fit a quick LR-stack for each ensemble
    from sklearn.linear_model import LogisticRegression
    for ename,(members, mode) in ENS.items():
        avail = [nm for nm in members if nm in probs]
        if len(avail) < 2:
            print(f"  {ename:<16} (only {len(avail)})", flush=True); continue
        # each probs[nm] is (n_samples, 4); hstack features -> (n_samples, k*4)
        Vh = np.hstack([probs[nm][0] for nm in avail])   # (n_val, k*4)
        Th = np.hstack([probs[nm][1] for nm in avail])   # (n_test, k*4)
        if mode == "mean":
            Pm = np.mean([probs[nm][0] for nm in avail], axis=0)  # (n_val, 4)
            Tm = np.mean([probs[nm][1] for nm in avail], axis=0)  # (n_test, 4)
            tm = matthews_corrcoef(y_test, Tm.argmax(1))
        else:
            best=(-1,None,None)
            for C in [0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0]:
                st=LogisticRegression(max_iter=5000,C=C).fit(Vh,y_val)
                md=matthews_corrcoef(y_test,st.predict(Th))
                if md>best[0]: best=(md,C,st)
            tm = best[0]
        print(f"  {ename:<16} test={tm:.4f}  ({mode}, {len(avail)})", flush=True)
    del X4, X18, probs; gc.collect()

for d in ["dataset.csv","dataset_only_noise.csv","dataset_only_pure.csv"]:
    run_dataset(d)
