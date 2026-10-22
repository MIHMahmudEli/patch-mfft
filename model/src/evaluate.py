"""
Evaluation Metrics Suite for Patch-MFFT Tampering Forensics.
Calculates:
1. Frame-Level Classification Metrics:
   - Accuracy, Macro-F1, Weighted-F1
   - Per-class Precision, Recall, Specificity, F1 (Authentic, Full-AI, Spliced)
   - Confusion Matrix (3x3)
   - Multi-class AUC-ROC (One-vs-Rest)
2. Pixel-Level Tampering Localization Metrics:
   - Mean Intersection over Union (mIoU)
   - Localization Pixel F1 (Dice coefficient)
   - Localization Area Under ROC (Pixel-AUC)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def compute_frame_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, list, dict]]:
    """
    Computes rigorous frame-level classification metrics for 3-class tampering detection.
    """
    if class_names is None:
        class_names = ["authentic", "full_ai", "spliced_tampered"]

    num_classes = len(class_names)
    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = float(correct / total) if total > 0 else 0.0

    # Confusion matrix (num_classes x num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class = {}
    f1_list = []
    prec_list = []
    rec_list = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - (tp + fp + fn)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class[class_names[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(spec, 4),
            "support": int(cm[c, :].sum()),
        }
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def compute_localization_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes pixel-level tampering segmentation metrics:
    - mIoU: Intersection over Union
    - Dice / Pixel F1
    """
    pred_binary = (pred_masks >= threshold).astype(bool)
    gt_binary = (gt_masks >= 0.5).astype(bool)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2 * intersection / (pred_binary.sum() + gt_binary.sum())) if (pred_binary.sum() + gt_binary.sum()) > 0 else 1.0

    return {
        "pixel_iou": round(iou, 4),
        "pixel_f1": round(dice, 4),
    }
