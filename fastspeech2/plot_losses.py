"""[new] Draw training_loss.png and validation_loss.png from the CSV logs written by trainer.py."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from trainer import LOSS_NAMES

COLORS = {
    "total": "#222222",
    "mel": "#2a6fdb",
    "mel_postnet": "#5aa9e6",
    "pitch": "#d6452b",
    "energy": "#7b2cbf",
    "duration": "#2b9348",
}


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in ["step"] + LOSS_NAMES}


def _plot(main, title, path, reference=None, reference_label=None):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5))
    if reference is not None:
        ax0.plot(reference["step"], reference["total"], color="#bbbbbb", lw=1, label=reference_label)
    ax0.plot(main["step"], main["total"], color=COLORS["total"], lw=1.6, label="total")
    ax0.set_title(title + ": total loss")
    if reference is not None:
        # the first training steps (loss > 10) would otherwise flatten the validation curve
        ax0.set_ylim(0, max(main["total"]) * 1.4)
    for name in LOSS_NAMES[1:]:
        ax1.plot(main["step"], main[name], color=COLORS[name], lw=1.4, label=name)
    ax1.set_title(title + ": components")
    ax1.set_yscale("log")
    for ax in (ax0, ax1):
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_losses(log_path, out_dir, language):
    os.makedirs(out_dir, exist_ok=True)
    train = read_csv(os.path.join(log_path, "train_loss.csv"))
    val_file = os.path.join(log_path, "val_loss.csv")
    val = read_csv(val_file) if os.path.exists(val_file) else None
    _plot(train, language + " training", os.path.join(out_dir, "training_loss.png"))
    if val is not None:
        _plot(
            val,
            language + " validation",
            os.path.join(out_dir, "validation_loss.png"),
            reference=train,
            reference_label="training total (for comparison)",
        )
    print("Saved loss plots to", out_dir)
