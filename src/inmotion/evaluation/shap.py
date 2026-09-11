#!/usr/bin/env python3
"""Game-Theoretic SHAP Feature Channel Attribution for WiFi ISAC Models.

Evaluates Shapley Additive exPlanations (SHAP) across the 18-channel feature hierarchy
(or 4-channel base hierarchy) for time-series world models (TS-JEPA, LeJEPA, CF-JEPA, T-JEPA, SIGReg).

Usage:
    python compute_shap.py --data data/processed/dataset.csv --checkpoint backup/ts_jepa_ft_seed42.pt --samples 50 --nsamples 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from inmotion.pipelines.mega_ensemble import (
    _build_cf_jepa,
    _build_lejepa,
    _build_sigreg,
    _build_t_jepa,
    _build_ts_jepa,
    _load_features,
    _stratified_indices,
)

FEATURE_NAMES_18 = [
    "Raw RSSI",
    r"Velocity ($\Delta$)",
    r"Acceleration ($\Delta^2$)",
    "Window Dev.",
    "Rolling FFT DC",
    "Rolling FFT AC",
    "Rolling Mean",
    "Rolling Range",
    "Rolling Std",
    "Rolling Skew",
    "Rolling Kurtosis",
    "Peak Location",
    "Slope Sign",
    "Sign Change Count",
    "STFT Band 0",
    "STFT Band 1",
    "STFT Band 2",
    "STFT Band 3",
]

FEATURE_FAMILIES_18 = [
    "Base", "Base", "Base", "Base",
    "Spectral", "Spectral",
    "Stats", "Stats", "Stats", "Stats", "Stats",
    "Shape", "Shape", "Shape",
    "Spectral", "Spectral", "Spectral", "Spectral",
]

FEATURE_NAMES_4 = [
    "Raw RSSI",
    r"Velocity ($\Delta$)",
    r"Acceleration ($\Delta^2$)",
    "Window Dev.",
]

MODEL_BUILDERS = {
    "ts_jepa": (_build_ts_jepa, True),
    "t_jepa": (_build_t_jepa, True),
    "cf_jepa": (_build_cf_jepa, True),
    "lejepa": (_build_lejepa, False),
    "sigreg": (_build_sigreg, False),
}


def explain_channels_shap(
    model: torch.nn.Module,
    X_test: np.ndarray,
    bg_channels: np.ndarray,
    n_channels: int,
    device: torch.device,
    num_test_samples: int = 50,
    nsamples_coalitions: int = 150,
) -> np.ndarray:
    """Compute channel-level Shapley values across test sequences.

    Args:
        model: PyTorch model in eval mode.
        X_test: Test feature array of shape (N_test, T, C).
        bg_channels: Empirical background channel values of shape (1, 1, C).
        n_channels: Number of feature channels (18 or 4).
        device: PyTorch device.
        num_test_samples: Number of test sequences to explain.
        nsamples_coalitions: Number of coalition evaluations per test sample.

    Returns:
        Array of Shapley attributions of shape (num_test_samples, n_channels, n_classes).
    """
    all_attributions = []
    eval_samples = X_test[:num_test_samples]

    for s in eval_samples:
        def model_fn(z: np.ndarray) -> np.ndarray:
            # z is (M, n_channels) binary coalition indicators
            M = z.shape[0]
            x_eval = np.tile(bg_channels, (M, 10, 1))
            for c in range(n_channels):
                mask = z[:, c] == 1
                if np.any(mask):
                    x_eval[mask, :, c] = s[:, c]
            with torch.no_grad():
                x_t = torch.from_numpy(x_eval.astype(np.float32)).to(device)
                out = model(x_t)
                if isinstance(out, tuple):
                    out = out[0]
                probs = torch.softmax(out, dim=-1).cpu().numpy()
            return probs

        explainer = shap.KernelExplainer(model_fn, np.zeros((1, n_channels)))
        # Explain the coalition z = 1 (all channels active)
        sv = explainer.shap_values(
            np.ones((1, n_channels)), nsamples=nsamples_coalitions, silent=True
        )
        all_attributions.append(sv)

    return np.array(all_attributions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Game-Theoretic SHAP Feature Channel Attribution."
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/dataset.csv"), help="Path to CSV dataset.")
    parser.add_argument(
        "--model", type=str, default="ts_jepa", choices=list(MODEL_BUILDERS.keys()), help="Model type."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("backup/ts_jepa_ft_seed42.pt"), help="Model checkpoint path."
    )
    parser.add_argument("--seed", type=int, default=42, help="Stratified split seed.")
    parser.add_argument("--samples", type=int, default=50, help="Number of test sequences to explain.")
    parser.add_argument(
        "--nsamples", type=int, default=150, help="Number of coalition evaluations in KernelExplainer."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="Directory for output CSV and plots."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    y_raw = pd.read_csv(args.data)["label"].values
    train_idx, _, test_idx = _stratified_indices(y_raw, args.seed)

    builder_fn, is_rich = MODEL_BUILDERS[args.model]
    n_channels = 18 if is_rich else 4
    feature_names = FEATURE_NAMES_18 if is_rich else FEATURE_NAMES_4
    feature_families = FEATURE_FAMILIES_18 if is_rich else ["Base"] * 4

    print(f"Loading features (rich={is_rich}, channels={n_channels})...")
    X = _load_features(args.data, rich=is_rich)

    # Build and load model
    print(f"Building {args.model} and loading weights from {args.checkpoint}...")
    model = builder_fn()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.model == "sigreg":
        sd = {"m." + k if not k.startswith("m.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # Background empirical channel distributions
    bg_channels = X[train_idx].mean(axis=(0, 1), keepdims=True)

    # Run SHAP
    print(f"Computing SHAP attributions for {args.samples} test samples (nsamples={args.nsamples})...")
    shap_matrix = explain_channels_shap(
        model=model,
        X_test=X[test_idx],
        bg_channels=bg_channels,
        n_channels=n_channels,
        device=device,
        num_test_samples=args.samples,
        nsamples_coalitions=args.nsamples,
    )

    # Global mean absolute SHAP per channel: mean over samples, timesteps/classes
    if shap_matrix.ndim == 4:
        mean_abs_shap = np.abs(shap_matrix).mean(axis=(0, 1, 3))
    else:
        mean_abs_shap = np.abs(shap_matrix).mean(axis=(0, 2))
    # Save to DataFrame
    df_shap = pd.DataFrame({
        "channel_idx": range(n_channels),
        "channel_name": feature_names,
        "family": feature_families,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = args.output_dir / f"shap_channel_attribution_{args.model}.csv"
    df_shap.to_csv(csv_path, index=False)
    print(f"\nSaved SHAP attribution table to {csv_path}")

    print("\n=== SHAP Feature Channel Ranking ===")
    for rank, (_, row) in enumerate(df_shap.iterrows()):
        fam = row["family"]
        name = row["channel_name"]
        val = row["mean_abs_shap"]
        print(f"  {rank+1:2d}. {name:25s} [{fam:8s}]: mean |SHAP| = {val:.5f}")

if __name__ == "__main__":
    main()
