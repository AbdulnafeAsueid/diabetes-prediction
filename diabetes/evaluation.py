"""Metrics and plots: classification report, confusion matrix, one-vs-rest ROC curves."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from .data import CLASS_NAMES, CLASSES


def evaluate(y_true, y_pred, y_score=None):
    """Accuracy, per-class report and (if probabilities are given) weighted one-vs-rest ROC-AUC."""
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }
    if y_score is not None:
        result["roc_auc_weighted_ovr"] = roc_auc_score(
            label_binarize(y_true, classes=CLASSES), y_score, multi_class="ovr", average="weighted")
    return result


def report_text(y_true, y_pred):
    return classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES,
                                 digits=4, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    labels = [f"{c} - {n}" for c, n in zip(CLASSES, CLASS_NAMES)]
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_roc(y_true, y_score, title, path):
    """One ROC curve per class (one-vs-rest) plus the micro average."""
    y_bin = label_binarize(y_true, classes=CLASSES)
    plt.figure(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, ["blue", "green", "red"])):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    fpr, tpr, _ = roc_curve(y_bin.ravel(), np.asarray(y_score).ravel())
    plt.plot(fpr, tpr, color="gray", lw=1, linestyle=":", label=f"Micro average (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_results(out_dir, name, y_true, y_pred, y_score=None):
    """Write metrics.json, report.txt, confusion_matrix.png and roc.png to out_dir/name/."""
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(y_true, y_pred, y_score)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    text = report_text(y_true, y_pred)
    (out / "report.txt").write_text(text)
    plot_confusion_matrix(y_true, y_pred, f"Confusion matrix - {name}", out / "confusion_matrix.png")
    if y_score is not None:
        plot_roc(y_true, y_score, f"ROC curves - {name}", out / "roc.png")
    return metrics, text
