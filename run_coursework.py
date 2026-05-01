from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score

from src.evaluate import EvalResult, build_models, fit_eval
from src.poisoning import (
    benign_label_index,
    poison_random,
    poison_systematic_benign,
    poison_targeted_borderline_attacks,
)
from src.preprocess import load_raw_csv, preprocess_dataframe, train_val_split_scaled


def attack_macro_recall(y_true: np.ndarray, y_pred: np.ndarray, benign_idx: int) -> float:
    labels = [c for c in np.unique(np.concatenate([y_true, y_pred])) if c != benign_idx]
    if not labels:
        return 1.0
    return float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def plot_model_bar(results: list[EvalResult], out_path: Path) -> None:
    names = [r.name for r in results]
    macro = [r.macro_f1 for r in results]
    plt.figure(figsize=(10, 4))
    plt.bar(names, macro, color="#2c7fb8")
    plt.ylabel("Macro F1 (test)")
    plt.xticks(rotation=25, ha="right")
    plt.title("Part A — model comparison (higher is better)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_poisoning_curves(records: list[dict], metric_key: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    strategies = sorted({r["strategy"] for r in records})
    for strat in strategies:
        xs = sorted({r["poison_fraction"] for r in records if r["strategy"] == strat})
        ys = [next(r[metric_key] for r in records if r["strategy"] == strat and r["poison_fraction"] == x) for x in xs]
        plt.plot([x * 100 for x in xs], ys, marker="o", linewidth=2, label=strat)
    plt.xlabel("Poisoned training labels (% of training rows)")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_all(
    data_path: Path,
    out_dir: Path | None = None,
    nrows: int | None = None,
    seed: int = 42,
    fast_models: bool = False,
) -> dict | None:
    """Execute Parts A and B. Returns a small summary dict, or None if data_path is missing."""
    if not data_path.is_file():
        print("CSV not found:", data_path.resolve())
        print("Download instructions: see DATASET.txt or run: python scripts/download_dataset.py")
        return None

    out_dir = out_dir or Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print("Loading CSV...")
    df = load_raw_csv(str(data_path), nrows=nrows)
    print("CSV loaded successfully")
    print("Rows:", len(df))
    print(df.head())
    print("Rows:", len(df), "Cols:", len(df.columns))

    if "Attack_type" in df.columns:
        eda = df["Attack_type"].astype(str).value_counts().rename_axis("Attack_type").reset_index(name="count")
        eda.to_csv(out_dir / "eda_attack_type_counts_raw.csv", index=False)

    print("Preprocessing...")
    X, y = preprocess_dataframe(df, random_state=seed)
    eda_post = y.astype(str).value_counts().rename_axis("Attack_type").reset_index(name="count_after_preprocess")
    eda_post.to_csv(out_dir / "eda_attack_type_counts_after_preprocess.csv", index=False)
    X_train, X_test, y_train, y_test, le, _scaler = train_val_split_scaled(X, y, random_state=seed)
    benign_idx = benign_label_index(le.classes_)
    labels = sorted(set(map(int, np.unique(np.concatenate([y_train, y_test])))))

    models = build_models(random_state=seed)
    if fast_models:
        models.pop("mlp_neural_net", None)
        models.pop("xgboost", None)
    part_a_rows: list[dict] = []
    eval_results: list[EvalResult] = []

    print("Part A — training and evaluation...")
    for name, model in models.items():
        print("  Model:", name)
        res = fit_eval(name, model, X_train, y_train, X_test, y_test, labels)
        eval_results.append(res)
        part_a_rows.append(
            {
                "model": name,
                "accuracy": res.accuracy,
                "macro_f1": res.macro_f1,
                "weighted_f1": res.weighted_f1,
            }
        )

    part_a_df = pd.DataFrame(part_a_rows).sort_values("macro_f1", ascending=False)
    part_a_df.to_csv(out_dir / "part_a_model_scores.csv", index=False)
    plot_model_bar(eval_results, out_dir / "part_a_macro_f1_bar.png")
    print(part_a_df.to_string(index=False))

    best_name = str(part_a_df.iloc[0]["model"])
    best_model = next(r.model for r in eval_results if r.name == best_name)
    print("Selected best model by macro F1:", best_name)

    best_pred = best_model.predict(X_test)
    y_test_bin = (y_test != benign_idx).astype(int)
    pred_bin = (best_pred != benign_idx).astype(int)
    plt.figure(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test_bin,
        pred_bin,
        display_labels=["Benign", "Attack"],
        colorbar=False,
    )
    plt.title("Part A — binary confusion (best model)")
    plt.tight_layout()
    plt.savefig(out_dir / "part_a_binary_confusion.png", dpi=160)
    plt.close()

    poison_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    strategies = {
        "random_label_flips": lambda yt, f: poison_random(
            yt, num_classes=len(le.classes_), poison_fraction=f, rng=rng
        ),
        "systematic_attack_to_benign": lambda yt, f: poison_systematic_benign(
            yt, benign_label=benign_idx, poison_fraction=f, rng=rng
        ),
        "targeted_borderline_attack_to_benign": lambda yt, f: poison_targeted_borderline_attacks(
            X_train, yt, benign_label=benign_idx, poison_fraction=f, rng=rng
        ),
    }

    part_b_records: list[dict] = []
    print("Part B — poisoning sweep on best model...")
    y_train_clean = np.asarray(y_train, dtype=np.int64).copy()
    for strat_name, poison_fn in strategies.items():
        for f in poison_fracs:
            y_tr = poison_fn(y_train_clean.copy(), f)
            num_changed = int(np.sum(y_tr != y_train_clean))
            m = clone(best_model)
            m.fit(X_train, y_tr)
            pred = m.predict(X_test)
            macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
            atk_recall = attack_macro_recall(y_test, pred, benign_idx)
            part_b_records.append(
                {
                    "strategy": strat_name,
                    "poison_fraction": float(f),
                    "num_labels_changed": num_changed,
                    "fraction_of_train_changed": float(num_changed / max(len(y_train_clean), 1)),
                    "macro_f1": macro_f1,
                    "attack_macro_recall": float(atk_recall),
                }
            )

    part_b_df = pd.DataFrame(part_b_records)
    baseline_f1 = float(part_b_df.loc[part_b_df["poison_fraction"] == 0.0, "macro_f1"].mean())
    part_b_df["f1_drop_vs_clean"] = baseline_f1 - part_b_df["macro_f1"]
    part_b_df["f1_drop_per_changed_label"] = part_b_df["f1_drop_vs_clean"] / part_b_df["num_labels_changed"].clip(
        lower=1
    )
    part_b_df.to_csv(out_dir / "part_b_poisoning_metrics.csv", index=False)

    plot_poisoning_curves(
        part_b_records,
        "macro_f1",
        "Part B — macro F1 vs poison rate (best Part A model)",
        out_dir / "part_b_macro_f1_vs_poison.png",
    )
    plot_poisoning_curves(
        part_b_records,
        "attack_macro_recall",
        "Part B — attack-class recall vs poison rate",
        out_dir / "part_b_attack_recall_vs_poison.png",
    )

    plt.figure(figsize=(8, 4.5))
    for strat in sorted(part_b_df["strategy"].unique()):
        sub = part_b_df[part_b_df["strategy"] == strat].sort_values("num_labels_changed")
        plt.plot(sub["num_labels_changed"], sub["macro_f1"], marker="o", linewidth=2, label=strat)
    plt.xlabel("Number of training labels changed")
    plt.ylabel("Macro F1 (test)")
    plt.title("Part B — macro F1 vs absolute number of flipped labels")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "part_b_macro_f1_vs_num_changed.png", dpi=160)
    plt.close()

    summary = {
        "best_model": best_name,
        "benign_class": str(le.classes_[benign_idx]),
        "n_features": int(X.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(le.classes_)),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote figures and tables to:", out_dir.resolve())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(Path("data") / "DNN-EdgeIIoT-dataset.csv"))
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for faster iteration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slower models (MLP, XGBoost) for quicker iteration on large CSVs.",
    )
    args = parser.parse_args()
    summary = run_all(
        Path(args.data),
        Path("outputs"),
        nrows=args.nrows,
        seed=args.seed,
        fast_models=args.fast,
    )
    return 0 if summary is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
