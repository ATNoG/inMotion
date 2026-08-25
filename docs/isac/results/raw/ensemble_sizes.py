"""Compute total parameters for every named ensemble (E1-E8).

Per-member param counts come from count_params.py (each checkpoint's
state_dict summed). Summed here per ensemble definition in the master file
(§4.0 registry). The LR-stack meta-learner is negligible ((n*4,4) weights).
"""
PARAMS = {
    "lejepa": 0.566, "t_jepa": 0.633, "ts_jepa": 2.153, "cf_jepa": 1.232,
    "sigreg": 0.697, "sigreg_s3": 0.697, "sigreg_s5": 0.697,
    "mamba3_cnn": 0.945, "mamba3_tcn": 0.843, "mamba3_transformer": 0.842,
    "mamba3_multiview": 1.701,
    "deepstack": 43.576, "cnn": 0.317, "tcn": 2.180, "gru": 0.039,
    "lstm": 0.052, "bilstm": 0.136,
    "hpo_gru": 4.008, "hpo_lstm": 13.110, "hpo_cnn": 0.081, "hpo_mamba": 0.278,
    "logreg": 0.00004,  # classical logistic regression on raw features, ~tiny
}

ENSEMBLES = {
    "E1 all-21 (icaisf)": ["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","sigreg_s3","sigreg_s5",
                           "mamba3_cnn","mamba3_tcn","mamba3_transformer","mamba3_multiview",
                           "hpo_gru","hpo_lstm","hpo_cnn","hpo_mamba",
                           "deepstack","cnn","tcn","gru","lstm","bilstm"],
    "E2 size-frontier": ["logreg","lejepa","sigreg"],
    "E3 16-members": ["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg",
                      "mamba3_cnn","mamba3_tcn","mamba3_transformer","mamba3_multiview",
                      "deepstack","cnn","tcn","hpo_gru","hpo_lstm","hpo_cnn","hpo_mamba"],
    "E4 config1 world-only": ["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","sigreg_s3","sigreg_s5"],
    "E5 config2 world+best-DL": ["lejepa","t_jepa","ts_jepa","cf_jepa","sigreg","sigreg_s3","sigreg_s5",
                                  "deepstack","mamba3_tcn"],
    "E6 config3 best-world": ["sigreg","lejepa","ts_jepa"],
    "E7 config4 best-world+best-DL": ["sigreg","lejepa","ts_jepa","deepstack","mamba3_tcn"],
    "E8 config6 greedy": ["lejepa","mamba3_multiview","mamba3_transformer","ts_jepa","sigreg","cf_jepa",
                          "mamba3_cnn","sigreg_s3","mamba3_tcn","t_jepa","hpo_lstm","sigreg_s5",
                          "deepstack","hpo_cnn","hpo_mamba","hpo_gru"],
}

print(f"{'Ensemble':<28} {'n':>3} {'Total params (M)':>17}")
print("-" * 50)
for name, members in ENSEMBLES.items():
    total = sum(PARAMS[m] for m in members)
    print(f"{name:<28} {len(members):>3} {total:>17.3f}")
