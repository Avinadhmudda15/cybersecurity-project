"""
Generate lab-style outputs (Weeks 1–10) under outputs/labs/.

Uses NumPy / scikit-learn / matplotlib only for deep-learning demos so it runs on Python 3.14
without TensorFlow. Place a photo at labs/car.jpg for Week 7; otherwise a synthetic image is used.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Crypto.Cipher import AES
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "labs"
LABS = ROOT / "labs"


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def week1_train_val_test_split() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    df_resampled = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "Attack Type": rng.choice(["Benign", "DDoS", "Scanning", "MITM"], n, p=[0.5, 0.2, 0.2, 0.1]),
        }
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        df_resampled.drop(columns=["Attack Type"]),
        df_resampled["Attack Type"],
        test_size=0.3,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )
    lines = [
        "Training Set: " + str(X_train.shape) + " " + str(y_train.shape),
        "Validation Set: " + str(X_val.shape) + " " + str(y_val.shape),
        "Testing Set: " + str(X_test.shape) + " " + str(y_test.shape),
    ]
    for ln in lines:
        print(ln)
    (OUT / "week1_split_shapes.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Week 1 ->", OUT / "week1_split_shapes.txt")


def week2_case_study() -> None:
    text = (
        "Week -2 Case study on Cyber-attacks (Twitter Bitcoin Scam, 2020): "
        "Attackers gained access to high-profile Twitter accounts and used them to promote a "
        "Bitcoin scam. Fake posts promised doubled returns; many users sent funds before Twitter "
        "intervened. The incident highlighted risks of centralized account recovery and weak "
        "internal controls."
    )
    (OUT / "week2_twitter_bitcoin_case_study.txt").write_text(text, encoding="utf-8")
    print("Week 2 ->", OUT / "week2_twitter_bitcoin_case_study.txt")


def week3_traffic_boxplot() -> None:
    rng = np.random.default_rng(7)
    base = rng.lognormal(mean=3.2, sigma=0.35, size=1800).clip(5, 80)
    outliers = rng.uniform(90, 150, size=40)
    vehicles = np.concatenate([base, outliers])
    df = pd.DataFrame({"Vehicles": vehicles})
    TARGET_COL = "Vehicles"
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df[TARGET_COL], ax=ax, color="#4c72b0")
    ax.set_title("Target Variable Box Plot")
    ax.set_ylabel("Vehicles")
    plt.tight_layout()
    fig.savefig(OUT / "week3_target_boxplot.png", dpi=160)
    plt.close(fig)
    print("Week 3 ->", OUT / "week3_target_boxplot.png")


def _load_mnist_subset(n_train: int = 8000, n_test: int = 1200) -> tuple[np.ndarray, ...]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = (mnist.data.astype(np.float32)[: n_train + n_test]) / 255.0
    y = mnist.target.astype(np.int64)[: n_train + n_test]
    X_train, X_test = X[:n_train], X[n_train : n_train + n_test]
    y_train, y_test = y[:n_train], y[n_train : n_train + n_test]
    return X_train, X_test, y_train, y_test


def _train_shallow_mlp(
    X: np.ndarray,
    y_oh: np.ndarray,
    hidden: int = 256,
    epochs: int = 12,
    lr: float = 0.15,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    k = y_oh.shape[1]
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((d, hidden)).astype(np.float32) * np.sqrt(2.0 / d)
    b1 = np.zeros((1, hidden), dtype=np.float32)
    W2 = rng.standard_normal((hidden, k)).astype(np.float32) * np.sqrt(2.0 / hidden)
    b2 = np.zeros((1, k), dtype=np.float32)
    idx = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            sel = idx[s : s + batch]
            xb = X[sel]
            yb = y_oh[sel]
            m = xb.shape[0]
            h_pre = xb @ W1 + b1
            h = np.maximum(0, h_pre)
            logits = h @ W2 + b2
            p = _softmax(logits)
            dlog = (p - yb) / max(m, 1)
            dW2 = h.T @ dlog
            db2 = np.sum(dlog, axis=0, keepdims=True)
            dh = (dlog @ W2.T) * (h_pre > 0).astype(np.float32)
            dW1 = xb.T @ dh
            db1 = np.sum(dh, axis=0, keepdims=True)
            W2 -= lr * dW2.astype(np.float32)
            b2 -= lr * db2.astype(np.float32)
            W1 -= lr * dW1.astype(np.float32)
            b1 -= lr * db1.astype(np.float32)
    return W1, b1, W2, b2


def _ce_grad_input(x: np.ndarray, y_idx: int, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    h_pre = x @ W1 + b1
    h = np.maximum(0, h_pre)
    logits = h @ W2 + b2
    p = _softmax(logits.reshape(1, -1)).ravel()
    yoh = np.zeros_like(p)
    yoh[y_idx] = 1.0
    dlog = p - yoh
    dh = (h_pre > 0).astype(np.float32) * (dlog @ W2.T)
    return dh @ W1.T


def week4_fgsm_mnist() -> None:
    X_train, X_test, y_train, y_test = _load_mnist_subset()
    ohe = OneHotEncoder(sparse_output=False, categories=[np.arange(10)])
    y_train_oh = ohe.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    W1, b1, W2, b2 = _train_shallow_mlp(X_train, y_train_oh, epochs=10, lr=0.12)

    def predict(x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ W1 + b1)
        return np.argmax(_softmax(h @ W2 + b2), axis=1)

    eps = 0.25
    n_show = 10
    rng = np.random.default_rng(1)
    pick = rng.choice(len(X_test), size=n_show, replace=False)
    originals = X_test[pick]
    labels = y_test[pick]
    noise = np.zeros_like(originals)
    adv = np.zeros_like(originals)
    for i in range(n_show):
        g = _ce_grad_input(originals[i], int(labels[i]), W1, b1, W2, b2)
        noise[i] = eps * np.sign(g)
        adv[i] = np.clip(originals[i] + noise[i], 0.0, 1.0)

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.2, 3.6))
    for j in range(n_show):
        axes[0, j].imshow(originals[j].reshape(28, 28), cmap="gray")
        axes[0, j].axis("off")
        axes[1, j].imshow(noise[j].reshape(28, 28), cmap="seismic")
        axes[1, j].axis("off")
        axes[2, j].imshow(adv[j].reshape(28, 28), cmap="gray")
        axes[2, j].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Noise", fontsize=9)
    axes[2, 0].set_ylabel("Adversarial", fontsize=9)
    plt.suptitle(f"FGSM (ε={eps}) on MNIST (NumPy MLP)")
    plt.tight_layout()
    fig.savefig(OUT / "week4_mnist_fgsm.png", dpi=160)
    plt.close(fig)
    clean_acc = accuracy_score(y_test, predict(X_test))
    X_adv = np.empty_like(X_test)
    for i in range(len(X_test)):
        g = _ce_grad_input(X_test[i], int(y_test[i]), W1, b1, W2, b2)
        X_adv[i] = np.clip(X_test[i] + eps * np.sign(g), 0.0, 1.0)
    adv_acc = accuracy_score(y_test, predict(X_adv))
    (OUT / "week4_fgsm_note.txt").write_text(
        f"Clean test accuracy (subset): {clean_acc:.3f}\n"
        f"Adversarial test accuracy (FGSM, same subset): {adv_acc:.3f}\n",
        encoding="utf-8",
    )
    print("Week 4 ->", OUT / "week4_mnist_fgsm.png")


def week5_gan_mnist_grid() -> None:
    """
    GAN generator output (MNIST) at Epoch 34.

    To match typical lab screenshots, this implements a small GAN (Generator + Discriminator)
    using NumPy only (no TensorFlow/PyTorch required).
    """
    # Keep this lightweight enough to run on typical laptops.
    X_train, _, _, _ = _load_mnist_subset(n_train=12_000, n_test=100)
    # Keep grayscale (closer to typical MNIST GAN screenshots), but increase contrast a bit.
    X_train = np.clip(X_train.astype(np.float32) ** 0.7, 0.0, 1.0)
    n, d = X_train.shape

    rng = np.random.default_rng(5)
    nz = 64
    h_g = 320
    h_d = 320
    batch = 256
    epochs = 34
    snapshot_epoch = 34

    def sigm(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

    def lrelu(x: np.ndarray, a: float = 0.2) -> np.ndarray:
        return np.where(x > 0, x, a * x)

    def dlrelu(x: np.ndarray, a: float = 0.2) -> np.ndarray:
        return np.where(x > 0, 1.0, a).astype(np.float32)

    # Generator: z -> hidden -> x (sigmoid)
    Wg1 = (rng.standard_normal((nz, h_g)).astype(np.float32) * 0.05)
    bg1 = np.zeros((1, h_g), dtype=np.float32)
    Wg2 = (rng.standard_normal((h_g, d)).astype(np.float32) * 0.05)
    bg2 = np.zeros((1, d), dtype=np.float32)

    # Discriminator: x -> hidden -> logit (sigmoid)
    Wd1 = (rng.standard_normal((d, h_d)).astype(np.float32) * 0.05)
    bd1 = np.zeros((1, h_d), dtype=np.float32)
    Wd2 = (rng.standard_normal((h_d, 1)).astype(np.float32) * 0.05)
    bd2 = np.zeros((1, 1), dtype=np.float32)

    # Adam optimizer state
    def adam_init(param: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(param), np.zeros_like(param)

    mWg1, vWg1 = adam_init(Wg1)
    mbg1, vbg1 = adam_init(bg1)
    mWg2, vWg2 = adam_init(Wg2)
    mbg2, vbg2 = adam_init(bg2)

    mWd1, vWd1 = adam_init(Wd1)
    mbd1, vbd1 = adam_init(bd1)
    mWd2, vWd2 = adam_init(Wd2)
    mbd2, vbd2 = adam_init(bd2)

    beta1, beta2 = 0.5, 0.999
    eps = 1e-8
    lr_g = 0.0012
    lr_d = 0.0006
    t = 0

    def adam_update(param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, lr: float, t_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1**t_step)
        vhat = v / (1.0 - beta2**t_step)
        param = param - lr * mhat / (np.sqrt(vhat) + eps)
        return param, m, v

    def G(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1 = z @ Wg1 + bg1
        h1 = np.tanh(a1)
        logits = h1 @ Wg2 + bg2
        x = sigm(logits)
        return x, h1, logits

    def D(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1 = x @ Wd1 + bd1
        h1 = lrelu(a1)
        logit = h1 @ Wd2 + bd2
        p = sigm(logit)
        return p, h1, a1

    def bce_grad_logit(p: np.ndarray, y: np.ndarray) -> np.ndarray:
        # For sigmoid + BCE: dL/dlogit = p - y
        return (p - y).astype(np.float32)

    def clip_grad(g: np.ndarray, clip: float = 1.0) -> np.ndarray:
        return np.clip(g, -clip, clip).astype(np.float32)

    z_show = rng.standard_normal((16, nz)).astype(np.float32)
    grid_epoch_time_s: float = 0.0
    final_grid: np.ndarray | None = None

    idx_all = np.arange(n)
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        rng.shuffle(idx_all)
        for s in range(0, n, batch):
            sel = idx_all[s : s + batch]
            x_real = X_train[sel]
            m = x_real.shape[0]
            if m == 0:
                continue

            # Train D once per step (faster; still stable with noise + smoothing).
            for _ in range(1):
                t += 1
                z = rng.standard_normal((m, nz)).astype(np.float32)
                x_fake, hg, _ = G(z)

                # Label smoothing for real, and a bit of noise
                y_real = np.full((m, 1), 0.9, dtype=np.float32)
                y_fake = np.zeros((m, 1), dtype=np.float32)

                # Instance noise helps early training and improves digit-like structure at low epochs.
                noise_std = 0.06 * (1.0 - ep / max(epochs, 1))
                x_real_n = np.clip(x_real + rng.normal(0, noise_std, size=x_real.shape).astype(np.float32), 0.0, 1.0)
                x_fake_n = np.clip(x_fake + rng.normal(0, noise_std, size=x_fake.shape).astype(np.float32), 0.0, 1.0)

                p_real, h_real, a_real = D(x_real_n)
                p_fake, h_fake, a_fake = D(x_fake_n)

                dlogit_real = bce_grad_logit(p_real, y_real) / max(m, 1)
                dlogit_fake = bce_grad_logit(p_fake, y_fake) / max(m, 1)

                dWd2 = h_real.T @ dlogit_real + h_fake.T @ dlogit_fake
                dbd2 = np.sum(dlogit_real + dlogit_fake, axis=0, keepdims=True)

                dh_real = (dlogit_real @ Wd2.T) * dlrelu(a_real)
                dh_fake = (dlogit_fake @ Wd2.T) * dlrelu(a_fake)
                dWd1 = x_real_n.T @ dh_real + x_fake_n.T @ dh_fake
                dbd1 = np.sum(dh_real + dh_fake, axis=0, keepdims=True)

                dWd2 = clip_grad(dWd2, 1.0)
                dbd2 = clip_grad(dbd2, 1.0)
                dWd1 = clip_grad(dWd1, 1.0)
                dbd1 = clip_grad(dbd1, 1.0)

                Wd2, mWd2, vWd2 = adam_update(Wd2, dWd2, mWd2, vWd2, lr_d, t)
                bd2, mbd2, vbd2 = adam_update(bd2, dbd2, mbd2, vbd2, lr_d, t)
                Wd1, mWd1, vWd1 = adam_update(Wd1, dWd1, mWd1, vWd1, lr_d, t)
                bd1, mbd1, vbd1 = adam_update(bd1, dbd1, mbd1, vbd1, lr_d, t)

            # Train G
            t += 1
            z = rng.standard_normal((m, nz)).astype(np.float32)
            x_fake, h_g1, logit_g = G(z)
            p_fake, h_d1, a_d1 = D(x_fake)
            y_trick = np.ones((m, 1), dtype=np.float32)  # want D(fake)=1
            dlogit = bce_grad_logit(p_fake, y_trick) / max(m, 1)

            # Backprop through D into x_fake
            dh = (dlogit @ Wd2.T) * dlrelu(a_d1)
            dx = dh @ Wd1.T

            # Backprop through sigmoid output of G
            dx = dx * (x_fake * (1.0 - x_fake))
            dWg2 = h_g1.T @ dx
            dbg2 = np.sum(dx, axis=0, keepdims=True)
            dhg = (dx @ Wg2.T) * (1.0 - h_g1**2)
            dWg1 = z.T @ dhg
            dbg1 = np.sum(dhg, axis=0, keepdims=True)

            dWg2 = clip_grad(dWg2, 1.0)
            dbg2 = clip_grad(dbg2, 1.0)
            dWg1 = clip_grad(dWg1, 1.0)
            dbg1 = clip_grad(dbg1, 1.0)

            Wg2, mWg2, vWg2 = adam_update(Wg2, dWg2, mWg2, vWg2, lr_g, t)
            bg2, mbg2, vbg2 = adam_update(bg2, dbg2, mbg2, vbg2, lr_g, t)
            Wg1, mWg1, vWg1 = adam_update(Wg1, dWg1, mWg1, vWg1, lr_g, t)
            bg1, mbg1, vbg1 = adam_update(bg1, dbg1, mbg1, vbg1, lr_g, t)

        ep_time = time.perf_counter() - t0
        if ep == snapshot_epoch:
            grid_epoch_time_s = float(ep_time)
            final_grid, _, _ = G(z_show)

    assert final_grid is not None

    # Make digits clearer for display (keep grayscale like typical GAN sample grids).
    final_grid = np.clip(final_grid ** 0.65, 0.0, 1.0).astype(np.float32)

    # Plot in a similar style to the reference screenshot.
    fig = plt.figure(figsize=(7.2, 6.0), facecolor="white")
    fig.suptitle("Week -5", y=0.98, fontsize=16, fontweight="bold")

    # Black panel background
    panel = fig.add_axes([0.08, 0.12, 0.84, 0.80])
    panel.set_facecolor("#1f1f1f")
    panel.set_xticks([])
    panel.set_yticks([])
    panel.set_frame_on(True)

    # White box for the grid
    box = fig.add_axes([0.18, 0.30, 0.44, 0.56])
    box.set_facecolor("white")
    box.set_xticks([])
    box.set_yticks([])
    box.set_frame_on(True)
    box.set_title("Epoch 34", fontsize=14, pad=10)

    # Draw 4x4 digits inside the white box
    pad_x, pad_y = 0.06, 0.08
    cell_w = (1.0 - 2 * pad_x) / 4.0
    cell_h = (1.0 - 2 * pad_y) / 4.0
    for i in range(16):
        r, c = divmod(i, 4)
        ax = box.inset_axes([pad_x + c * cell_w + 0.01, 1.0 - pad_y - (r + 1) * cell_h + 0.01, cell_w - 0.02, cell_h - 0.02])
        ax.imshow(final_grid[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    panel.text(
        0.03,
        0.05,
        f"Time for epoch 34 is {grid_epoch_time_s:.2f} sec",
        color="#dddddd",
        fontsize=10,
        fontfamily="monospace",
        transform=panel.transAxes,
    )

    out_path = OUT / "week5_gan_epoch34.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("Week 5 ->", out_path)


def week6_train_val_loss() -> None:
    X_train, X_test, y_train, y_test = _load_mnist_subset(n_train=5000, n_test=1000)
    le = LabelEncoder()
    y_train_e = le.fit_transform(y_train)
    y_test_e = le.transform(y_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=1,
        warm_start=True,
        random_state=42,
        learning_rate_init=0.001,
        batch_size=200,
    )
    n_epochs = 10
    train_losses: list[float] = []
    val_losses: list[float] = []
    for _ in range(n_epochs):
        mlp.partial_fit(X_train, y_train_e, classes=np.arange(10))
        if hasattr(mlp, "loss_") and mlp.loss_ is not None:
            train_losses.append(float(mlp.loss_))
        else:
            train_losses.append(0.0)
        prob = mlp.predict_proba(X_test)
        ll = -np.mean(np.log(prob[np.arange(len(y_test_e)), y_test_e] + 1e-12))
        val_losses.append(float(ll))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(train_losses)), train_losses, label="Training Loss", color="#1f77b4")
    ax.plot(range(len(val_losses)), val_losses, label="Validation Loss", color="#ff7f0e")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT / "week6_loss_curve.png", dpi=160)
    plt.close(fig)
    print("Week 6 ->", OUT / "week6_loss_curve.png")


def week7_aes_ecb_image() -> None:
    key = b"0123456789abcdef"
    car_path = LABS / "car.jpg"
    if car_path.is_file():
        img = Image.open(car_path).convert("L")
    else:
        # Create a high-contrast "scene" with repeated structures so ECB leakage is obvious.
        # (ECB is easiest to demonstrate when the plaintext has repeating blocks.)
        w0, h0 = 256, 252
        y, x = np.ogrid[:h0, :w0]

        # Sky gradient
        sky = (210 - (y * 60 / max(h0 - 1, 1))).astype(np.float32) + (0 * x).astype(np.float32)
        sky = np.clip(sky, 140, 220)

        # Road base
        road = (70 + (y * 40 / max(h0 - 1, 1))).astype(np.float32) + (0 * x).astype(np.float32)
        road = np.clip(road, 60, 120)

        arr = sky.copy()
        horizon = int(h0 * 0.55)
        arr[horizon:, :] = road[horizon:, :]

        # Repeating brick-like pattern on the road (ECB-friendly)
        tile_h, tile_w = 12, 16
        rr = ((y - horizon) // tile_h) % 2
        cc = (x // tile_w) % 2
        checker = (rr ^ cc).astype(np.float32)
        pattern = (checker * 25.0)  # contrast
        arr[horizon:, :] = np.clip(arr[horizon:, :] + pattern[horizon:, :], 0, 255)

        # Lane markings (repeating dashed rectangles)
        for x0 in (96, 128, 160):
            for y0 in range(horizon + 6, h0 - 6, 18):
                arr[y0 : y0 + 8, x0 : x0 + 6] = 230

        # Car silhouette (simple) with repeated windows to emphasize block repetition
        car_y = int(h0 * 0.68)
        car_x = 60
        car_w = 140
        car_h = 44
        arr[car_y : car_y + car_h, car_x : car_x + car_w] = 55
        # Roof
        arr[car_y - 18 : car_y, car_x + 28 : car_x + car_w - 28] = 65
        # Windows (repeating)
        for i in range(5):
            wx0 = car_x + 32 + i * 18
            arr[car_y - 14 : car_y - 4, wx0 : wx0 + 10] = 165
        # Wheels
        for cx in (car_x + 28, car_x + car_w - 36):
            for dy in range(-10, 10):
                for dx in range(-10, 10):
                    if dx * dx + dy * dy <= 90:
                        yy = car_y + car_h + dy - 2
                        xx = cx + dx
                        if 0 <= yy < h0 and 0 <= xx < w0:
                            arr[yy, xx] = 20

        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    w, h_img = 256, 252
    img = img.resize((w, h_img))
    raw = img.tobytes()
    assert len(raw) % AES.block_size == 0
    cipher = AES.new(key, AES.MODE_ECB)
    enc = cipher.encrypt(raw)
    enc_img = Image.frombytes("L", (w, h_img), enc)
    img.save(OUT / "week7_original.png")
    enc_img.save(OUT / "encrypted_ecb.png")
    (OUT / "week7_caption.txt").write_text(
        "DES stands for Data Encryption Standard. AES stands for Advanced Encryption Standard.\n"
        "ECB mode encrypts each block independently; patterns in the plaintext can leak into the ciphertext.\n",
        encoding="utf-8",
    )
    print("Week 7 ->", OUT / "encrypted_ecb.png")


def week8_diffie_hellman() -> None:
    partner = "Neyna"
    p, g = 23, 5
    a = 6
    b = 15
    A = pow(g, a, p)
    B = pow(g, b, p)
    s1 = pow(B, a, p)
    s2 = pow(A, b, p)
    lines = [
        f"Partner Name: {partner}",
        f"p = {p}, g = {g}, private a = {a} (peer private b = {b})",
        f"Public A = g^a mod p = {A}",
        f"Public B = g^b mod p = {B}",
        f"Shared secret (B^a mod p) = {s1}",
        f"Shared secret (A^b mod p) = {s2}",
        f"Match: {s1 == s2}",
    ]
    (OUT / "week8_diffie_hellman.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Week 8 ->", OUT / "week8_diffie_hellman.txt")


def week9_pow_chart() -> None:
    def mine_numeric(threshold_prefix: str = "0000", max_tries: int = 80_000) -> float:
        t0 = time.perf_counter()
        for i in range(max_tries):
            h = hashlib.sha256(f"block-{i}".encode()).hexdigest()
            if h.startswith(threshold_prefix):
                return time.perf_counter() - t0
        return time.perf_counter() - t0

    def mine_pattern() -> float:
        t0 = time.perf_counter()
        for i in range(400):
            h = hashlib.sha256(f"p-{i}".encode()).hexdigest()
            if "face" in h:
                return time.perf_counter() - t0
        return 0.0

    def mine_timelimited(seconds: float = 0.05) -> float:
        t0 = time.perf_counter()
        c = 0
        while time.perf_counter() - t0 < seconds:
            hashlib.sha256(str(c).encode()).hexdigest()
            c += 1
        return seconds

    t_num = mine_numeric("0000", 120_000)
    t_pat = mine_pattern()
    t_time = mine_timelimited(0.02)

    names = ["Numeric Threshold", "Pattern Match", "Time-limited"]
    vals = [t_num, t_pat, t_time]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color=["#2c7fb8", "#fdae61", "#a6d96a"])
    ax.set_ylabel("Mining Time (s)")
    ax.set_title("Comparison of Proof-of-Work Methods")
    plt.tight_layout()
    fig.savefig(OUT / "week9_pow_comparison.png", dpi=160)
    plt.close(fig)
    print("Week 9 ->", OUT / "week9_pow_comparison.png")


def week10_metrics_grouped_bar() -> None:
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
    from sklearn.ensemble import RandomForestClassifier

    models = {
        "Model A": RandomForestClassifier(n_estimators=40, max_depth=4, random_state=1),
        "Model B": RandomForestClassifier(n_estimators=80, max_depth=6, random_state=2),
        "Model C": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=3),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        rows.append(
            {
                "name": name,
                "accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred, average="macro", zero_division=0),
                "recall": recall_score(y_te, pred, average="macro", zero_division=0),
                "f1": f1_score(y_te, pred, average="macro", zero_division=0),
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - 1.5 * w, df["accuracy"], width=w, label="Accuracy", color="#1f77b4")
    ax.bar(x - 0.5 * w, df["precision"], width=w, label="Precision", color="#ff7f0e")
    ax.bar(x + 0.5 * w, df["recall"], width=w, label="Recall", color="#2ca02c")
    ax.bar(x + 1.5 * w, df["f1"], width=w, label="F1 Score", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparison of Accuracy, Precision, Recall and F1 Score")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "week10_model_metrics.png", dpi=160)
    plt.close(fig)
    print("Week 10 ->", OUT / "week10_model_metrics.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    week1_train_val_test_split()
    week2_case_study()
    week3_traffic_boxplot()
    week4_fgsm_mnist()
    week5_gan_mnist_grid()
    week6_train_val_loss()
    week7_aes_ecb_image()
    week8_diffie_hellman()
    week9_pow_chart()
    week10_metrics_grouped_bar()
    print("All lab artifacts written to:", OUT.resolve())


if __name__ == "__main__":
    main()
