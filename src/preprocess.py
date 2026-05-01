from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_raw_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Attack_type" in df.columns:
        return df
    for c in df.columns:
        if str(c).strip().lower().replace(" ", "_") in ("attack_type", "attacktype", "attack_label"):
            return df.rename(columns={c: "Attack_type"})
    raise ValueError("Expected a target column such as 'Attack_type' (Edge-IIoTset DNN CSV).")


def preprocess_dataframe(df: pd.DataFrame, random_state: int = 42) -> tuple[np.ndarray, pd.Series]:
    """Return feature matrix X (float32) and target Series y (string labels)."""
    del random_state  # reserved for future resampling hooks
    work = _normalize_target_column(df.copy())
    work = work.dropna(how="any")
    work = work.drop_duplicates()
    y = work["Attack_type"].astype(str)
    X_df = work.drop(columns=["Attack_type"])

    drop_names = {
        "attack_label",
        "label",
        "timestamp",
        "frame.time",
        "flow id",
        "attack_name",
    }
    for c in list(X_df.columns):
        cl = str(c).strip().lower()
        if cl in drop_names or "ip.src" in cl or "ip.dst" in cl:
            X_df = X_df.drop(columns=[c])

    for c in list(X_df.columns):
        col = X_df[c]
        if col.dtype == object or col.dtype == "string":
            nu = col.nunique(dropna=True)
            if nu > min(256, max(32, len(X_df) // 25)):
                X_df = X_df.drop(columns=[c])

    X_enc = pd.get_dummies(X_df, dummy_na=False, dtype=np.float32)
    if X_enc.shape[1] == 0:
        raise ValueError("No feature columns remain after preprocessing.")
    Xmat = np.ascontiguousarray(X_enc.to_numpy(dtype=np.float32))
    return Xmat, y.reset_index(drop=True)


def train_val_split_scaled(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, le, scaler
