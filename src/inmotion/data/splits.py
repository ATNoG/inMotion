"""The train/validation/test split convention.

One definition, because every reported number depends on it. `MASTER_RESULTS.md`
states the convention: an 80/20 stratified train-test split, then 12.5% of the
train portion held out for validation, both with a fixed seed.

This lived in `pipelines/mega_ensemble.py` and was imported from there by three
other modules, which meant a 1,000-line pipeline had to be imported to obtain a
fifteen-line function. It is a data concern, so it lives with the data.
"""

from __future__ import annotations

import numpy as np

#: Fraction of the data held out for testing.
TEST_SIZE = 0.2
#: Fraction of the *remaining* training portion held out for validation.
VALIDATION_SIZE = 0.125


def stratified_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train, validation, test)`` index arrays.

    Stratified by label so every split sees all classes, with the seed threaded
    through both steps so the result is a pure function of ``(y, seed)``.
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VALIDATION_SIZE,
        random_state=seed,
        stratify=y[train_val_idx],
    )
    return train_idx, val_idx, test_idx
