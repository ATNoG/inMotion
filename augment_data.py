"""Data Augmentation for RSSI sequences.

Two strategies:
  1. Class-Conditioned Noise Augmentation
       - noise=False (clean): inject heavier Gaussian noise to teach model
         how "clean" data looks when it gets dirty.
       - noise=True  (noisy): apply only light jitter so the real signal
         is not overwhelmed.
  2. Intra-class Mixup
       λ ~ Beta(α, α),  X_new = λ·X₁ + (1-λ)·X₂   (same label only)

All synthetic rows get  synthetic=True; originals get synthetic=False.
"""

import numpy as np
import pandas as pd

# ── configuration ────────────────────────────────────────────────────────────
RSSI_COLS = [str(i) for i in range(1, 11)]  # columns "1" … "10"
SEED = 42

# Noise augmentation
CLEAN_NOISE_STD = 3.0  # dBm  – heavier noise for noise=False rows
CLEAN_COPIES = 6  # synthetic copies per clean sample
NOISY_NOISE_STD = 0.8  # dBm  – light jitter for noise=True rows
NOISY_COPIES = 4  # synthetic copies per noisy sample

# Mixup
MIXUP_ALPHA = 0.4  # Beta(α, α) – values near 0/1 keep samples realistic
MIXUP_PER_CLASS = 600  # mixup pairs to generate per class label
# ─────────────────────────────────────────────────────────────────────────────


def noise_augment(
    df: pd.DataFrame,
    std: float,
    n_copies: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a DataFrame of *n_copies* noisy versions of every row in df."""
    rssi = df[RSSI_COLS].values  # (N, 10)
    rows = []
    for _ in range(n_copies):
        jitter = rng.normal(0.0, std, size=rssi.shape)
        new_rssi = np.round(rssi + jitter, 1)
        chunk = df.copy()
        chunk[RSSI_COLS] = new_rssi
        chunk["synthetic"] = True
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def mixup_augment(
    df: pd.DataFrame,
    n_per_class: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Intra-class Mixup: X_new = λ·X₁ + (1−λ)·X₂, same label only."""
    synthetic_rows = []
    for label, group in df.groupby("label"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue  # need at least 2 samples

        lambdas = rng.beta(alpha, alpha, size=n_per_class)
        idx1 = rng.integers(0, n, size=n_per_class)
        idx2 = rng.integers(0, n, size=n_per_class)
        # avoid trivial self-mix
        same = idx1 == idx2
        idx2[same] = (idx2[same] + 1) % n

        rssi1 = group.loc[idx1, RSSI_COLS].values
        rssi2 = group.loc[idx2, RSSI_COLS].values
        lam = lambdas[:, None]  # (n_per_class, 1) for broadcasting
        mixed_rssi = np.round(lam * rssi1 + (1 - lam) * rssi2, 1)

        # Inherit meta columns from sample 1
        meta = group.loc[idx1].reset_index(drop=True).copy()
        meta[RSSI_COLS] = mixed_rssi
        meta["synthetic"] = True
        synthetic_rows.append(meta)

    return pd.concat(synthetic_rows, ignore_index=True)


def augment(
    input_path: str = "dataset.csv", output_path: str = "dataset_augmented.csv"
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(input_path)
    df["synthetic"] = False

    # ── 1. Class-Conditioned Noise Augmentation ───────────────────────────
    clean_df = df[df["noise"] == False]
    noisy_df = df[df["noise"] == True]

    print(f"Original rows  : {len(df)}")
    print(f"  noise=False  : {len(clean_df)}")
    print(f"  noise=True   : {len(noisy_df)}")

    aug_clean = noise_augment(clean_df, CLEAN_NOISE_STD, CLEAN_COPIES, rng)
    aug_noisy = noise_augment(noisy_df, NOISY_NOISE_STD, NOISY_COPIES, rng)

    print(f"\nNoise-augmented rows (clean→dirty) : {len(aug_clean)}")
    print(f"Noise-augmented rows (noisy light) : {len(aug_noisy)}")

    # ── 2. Mixup (on original rows only, intra-class) ─────────────────────
    aug_mixup = mixup_augment(df, MIXUP_PER_CLASS, MIXUP_ALPHA, rng)
    print(f"Mixup rows                         : {len(aug_mixup)}")

    # ── 3. Combine ────────────────────────────────────────────────────────
    combined = pd.concat([df, aug_clean, aug_noisy, aug_mixup], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nFinal dataset rows : {len(combined)}")
    print(f"  synthetic=False  : {(combined['synthetic'] == False).sum()}")
    print(f"  synthetic=True   : {(combined['synthetic'] == True).sum()}")
    print("\nLabel distribution (augmented):")
    print(combined["label"].value_counts())

    combined.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    return combined


if __name__ == "__main__":
    augment()
