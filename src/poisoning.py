from __future__ import annotations

import numpy as np


def benign_label_index(classes: np.ndarray) -> int:
    for i, c in enumerate(np.asarray(classes)):
        if "benign" in str(c).lower():
            return int(i)
    return 0


def poison_random(
    y: np.ndarray,
    num_classes: int,
    poison_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).copy()
    n = len(y)
    k = int(round(n * float(poison_fraction)))
    if k <= 0 or num_classes < 2:
        return y
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        choices = [c for c in range(num_classes) if c != int(y[i])]
        y[i] = int(rng.choice(choices))
    return y


def poison_systematic_benign(
    y: np.ndarray,
    benign_label: int,
    poison_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).copy()
    attack_idx = np.where(y != benign_label)[0]
    if len(attack_idx) == 0:
        return y
    rng.shuffle(attack_idx)
    k = int(round(len(attack_idx) * float(poison_fraction)))
    y[attack_idx[:k]] = benign_label
    return y


def poison_targeted_borderline_attacks(
    X_train: np.ndarray,
    y: np.ndarray,
    benign_label: int,
    poison_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    del rng
    y = np.asarray(y, dtype=np.int64).copy()
    benign_mask = y == benign_label
    if not np.any(benign_mask):
        return poison_systematic_benign(y, benign_label, poison_fraction, np.random.default_rng(0))
    centroid = X_train[benign_mask].mean(axis=0)
    attack_idx = np.where(y != benign_label)[0]
    if len(attack_idx) == 0:
        return y
    dists = np.linalg.norm(X_train[attack_idx] - centroid, axis=1)
    order = np.argsort(dists)
    k = int(round(len(attack_idx) * float(poison_fraction)))
    flip_idx = attack_idx[order[:k]]
    y[flip_idx] = benign_label
    return y
