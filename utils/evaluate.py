
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize


def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def print_classification_report(labels, preds, class_names, dataset_name):
    acc = accuracy_score(labels, preds) * 100
    p   = precision_score(labels, preds, average="weighted") * 100
    r   = recall_score(labels, preds,    average="weighted") * 100
    f1  = f1_score(labels, preds,        average="weighted") * 100
    print(f"\n{'='*55}")
    print(f"  Classification Report — {dataset_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {p:.2f}%")
    print(f"  Recall    : {r:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print(f"{'='*55}")
    print(classification_report(labels, preds, target_names=class_names))
    return acc, p, r, f1


def plot_confusion_matrix(labels, preds, class_names,
                           dataset_name, save_dir="results/evaluation"):
    os.makedirs(save_dir, exist_ok=True)
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Confusion Matrix — {dataset_name}", fontsize=15)
    for ax, data, title in zip(axes,
                                [cm, cm_norm],
                                ["Raw Counts", "Normalized (%)"]):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        thresh = data.max() / 2
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                txt = str(int(val)) if data is cm else f"{val:.1f}%"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=9)
    plt.tight_layout()
    path = f"{save_dir}/{dataset_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved to {path}")


def plot_roc_curves(labels, probs, class_names,
                    dataset_name, save_dir="results/evaluation"):
    os.makedirs(save_dir, exist_ok=True)
    n_classes  = len(class_names)
    labels_bin = label_binarize(labels, classes=range(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])
    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr  /= n_classes
    macro_auc  = auc(all_fpr, mean_tpr)
    fig, axes  = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"ROC Curves — {dataset_name}", fontsize=15)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        axes[0].plot(fpr[i], tpr[i], color=color, lw=1.5,
                     label=f"{cls} (AUC={roc_auc[i]:.2f})")
    axes[0].plot([0,1],[0,1], "k--", lw=1, label="Random (0.50)")
    axes[0].plot(all_fpr, mean_tpr, "b-", lw=2.5,
                 label=f"Macro avg (AUC={macro_auc:.2f})")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve per Class")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    sorted_idx = np.argsort([roc_auc[i] for i in range(n_classes)])
    sorted_cls = [class_names[i] for i in sorted_idx]
    sorted_auc = [roc_auc[i]     for i in sorted_idx]
    bars = axes[1].barh(sorted_cls, sorted_auc,
                         color=[colors[i] for i in sorted_idx])
    axes[1].axvline(x=macro_auc, color="blue", linestyle="--",
                    lw=1.5, label=f"Macro={macro_auc:.2f}")
    axes[1].axvline(x=0.5, color="gray", linestyle="--",
                    lw=1, label="Random=0.50")
    axes[1].set_xlim([0.0, 1.05])
    axes[1].set_xlabel("AUC Score")
    axes[1].set_title("AUC Score per Class")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, sorted_auc):
        axes[1].text(val+0.01, bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = f"{save_dir}/{dataset_name}_roc_auc.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved to {path}")
    print(f"Macro AUC: {macro_auc:.4f}")
    return roc_auc, macro_auc


def plot_precision_recall(labels, preds, class_names,
                           dataset_name, save_dir="results/evaluation"):
    os.makedirs(save_dir, exist_ok=True)
    precision = precision_score(labels, preds, average=None) * 100
    recall    = recall_score(labels, preds,    average=None) * 100
    f1        = f1_score(labels, preds,        average=None) * 100
    x, w      = np.arange(len(class_names)), 0.25
    fig, ax   = plt.subplots(figsize=(14, 6))
    ax.bar(x-w, precision, w, label="Precision", color="steelblue",  alpha=0.85)
    ax.bar(x,   recall,    w, label="Recall",    color="darkorange", alpha=0.85)
    ax.bar(x+w, f1,        w, label="F1 Score",  color="green",      alpha=0.85)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Precision Recall F1 — {dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 115])
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i-w, p+1, f"{p:.0f}%", ha="center", fontsize=8)
        ax.text(i,   r+1, f"{r:.0f}%", ha="center", fontsize=8)
        ax.text(i+w, f+1, f"{f:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    path = f"{save_dir}/{dataset_name}_precision_recall.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved to {path}")


def evaluate_full(model, test_loader, class_names,
                   dataset_name, device,
                   save_dir="results/evaluation"):
    print(f"\nRunning full evaluation for {dataset_name}...")
    labels, preds, probs = get_predictions(model, test_loader, device)
    acc, p, r, f1 = print_classification_report(
        labels, preds, class_names, dataset_name)
    plot_confusion_matrix(labels, preds, class_names,
                           dataset_name, save_dir)
    roc_auc, macro_auc = plot_roc_curves(
        labels, probs, class_names, dataset_name, save_dir)
    plot_precision_recall(labels, preds, class_names,
                           dataset_name, save_dir)
    return {"accuracy": acc, "precision": p,
            "recall": r, "f1": f1, "macro_auc": macro_auc}
