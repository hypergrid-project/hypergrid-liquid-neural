"""
HyperGrid Continuous-Time Deep Learning Pipeline
Component C: Training & Regularization Pipeline for Liquid Neural Networks (CfC)
=================================================================================
Features:
- Stratified Group K-Fold Cross-Validation (split by driving session/trip ID to prevent data leakage).
- Class-Weighted Cross-Entropy and Multi-Class Focal Loss to combat extreme imbalance.
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Early Stopping monitoring validation Macro-averaged F1-Score.
- Comprehensive metric tracking:
  * Multiclass Confusion Matrix
  * Macro-averaged Precision, Recall, and F1-score
  * False Positive Rate (FPR) on Normal Road segments (prevents driver alert fatigue)
- Automatic checkpointing and production deployment artifact serialization.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from utils.dataset_lnn import RobustSensorScaler, create_dataloader, load_hypergrid_dataset
from utils.model_cfc import CfCLiquidNet, build_cfc_model


# =============================================================================
# Loss Functions: Multi-Class Focal Loss
# =============================================================================

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss to counteract extreme class imbalance (e.g. rare potholes vs frequent smooth asphalt).
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, num_classes) logits
        # targets: (batch_size,) integer class labels
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = torch.pow(1.0 - target_p, self.gamma)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_factor = self.alpha.gather(dim=-1, index=targets)
            focal_loss = -alpha_factor * focal_weight * target_log_p
        else:
            focal_loss = -focal_weight * target_log_p

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(y: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """
    Compute balanced inverse-frequency weights:
    w_c = N / (C * N_c)
    """
    counts = np.bincount(y, minlength=num_classes)
    total_samples = len(y)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total_samples / (num_classes * counts[c])
        else:
            weights[c] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Metric Calculation Utilities
# =============================================================================

def compute_telemetry_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Computes rigorous telemetry validation metrics:
    - Macro F1, Precision, Recall
    - Normal Road False Positive Rate (FPR):
      FPR_normal = False Positives on normal road / (False Positives + True Negatives)
    - Confusion Matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Class 0: Normal Smooth Road
    # Anomaly false alarm is when true is 0, but pred is 1, 2, or 3
    normal_total = np.sum(y_true == 0)
    if normal_total > 0:
        normal_false_alarms = np.sum((y_true == 0) & (y_pred != 0))
        normal_fpr = float(normal_false_alarms) / float(normal_total)
    else:
        normal_fpr = 0.0

    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "normal_road_fpr": float(normal_fpr),
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# Epoch Trainer & Evaluator
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_x, batch_dt, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_dt = batch_dt.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x, batch_dt)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping for recurrent ODE stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_dt, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_dt = batch_dt.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x, batch_dt)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    metrics = compute_telemetry_metrics(all_targets_arr, all_preds_arr, num_classes=num_classes)
    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss, metrics, all_preds_arr, all_targets_arr


# =============================================================================
# Complete Training Runner
# =============================================================================

def train_cfc_pipeline(
    data_dir: str = "./data",
    output_dir: str = "./models/cfc",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 32,
    sequence_length: int = 100,
    stride: int = 30,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    k_folds: int = 5,
    device_name: str = "auto",
    use_conv_frontend: bool = True,
    patience: int = 10,
) -> Dict[str, any]:
    """
    Executes disciplined training with Stratified GroupKFold cross-validation.
    Saves best model weights, scaler, and training summary metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_name)
    print(f"[*] Executing CfC training on compute device: {device}")

    # 1. Load Data
    print(f"[*] Ingesting raw telemetry windows from '{data_dir}' (seq_len={sequence_length}, stride={stride})...")
    X, dt, y, groups = load_hypergrid_dataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride,
        nominal_hz=5.0,
        include_magnitudes=True,
        synthesize_bumps=True,
        bump_ratio=0.25,
    )
    num_samples, _, in_channels = X.shape
    num_classes = 4
    class_counts = np.bincount(y, minlength=num_classes)
    print(f" -> Total Windows: {num_samples} across {len(np.unique(groups))} trips")
    print(f" -> Class Distribution: Class 0 (Smooth): {class_counts[0]}, Class 1 (Pothole): {class_counts[1]}, "
          f"Class 2 (Speed Bump): {class_counts[2]}, Class 3 (Rough Road): {class_counts[3]}")

    # 2. Stratified GroupKFold setup
    sgkf = StratifiedGroupKFold(n_splits=k_folds)
    fold_metrics = []

    best_overall_f1 = -1.0
    best_model_path = os.path.join(output_dir, "best_cfc_model.pt")
    scaler_path = os.path.join(output_dir, "scaler_params.json")

    print("\n" + "=" * 65)
    print(f"   STARTING {k_folds}-FOLD STRATIFIED GROUP CROSS-VALIDATION")
    print("=" * 65)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n--- [Fold {fold + 1}/{k_folds}] ---")
        train_trips = set(groups[train_idx])
        val_trips = set(groups[val_idx])
        print(f" -> Train Samples: {len(train_idx)} ({len(train_trips)} trips), Val Samples: {len(val_idx)} ({len(val_trips)} trips)")

        X_train, dt_train, y_train = X[train_idx], dt[train_idx], y[train_idx]
        X_val, dt_val, y_val = X[val_idx], dt[val_idx], y[val_idx]

        # Fit Scaler on Train Split Only (Strict anti-leakage)
        scaler = RobustSensorScaler(method="robust")
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_val_norm = scaler.transform(X_val)

        # Build DataLoaders
        train_loader = create_dataloader(X_train_norm, dt_train, y_train, batch_size=batch_size, shuffle=True, augment=True)
        val_loader = create_dataloader(X_val_norm, dt_val, y_val, batch_size=batch_size, shuffle=False, augment=False)

        # Model instantiation
        model = CfCLiquidNet(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_classes=num_classes,
            use_conv_frontend=use_conv_frontend,
            conv_out_channels=16,
            mode="default",
            pooling="attention",
            dropout_p=0.2,
        ).to(device)

        # Loss Function
        class_weights = compute_class_weights(y_train, num_classes=num_classes).to(device)
        if loss_type == "focal":
            criterion = MultiClassFocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_f1 = -1.0
        best_val_metrics = {}
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, num_classes=num_classes)
            scheduler.step()

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_val_metrics = val_metrics
                no_improve_epochs = 0

                # Checkpoint global best
                if best_val_f1 > best_overall_f1:
                    best_overall_f1 = best_val_f1
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "in_channels": in_channels,
                            "hidden_size": hidden_size,
                            "num_classes": num_classes,
                            "use_conv_frontend": use_conv_frontend,
                            "sequence_length": sequence_length,
                            "best_f1": best_overall_f1,
                        },
                        best_model_path,
                    )
                    scaler.save_json(scaler_path)
            else:
                no_improve_epochs += 1

            if epoch % 5 == 0 or epoch == epochs or no_improve_epochs == 0:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
                    f"Normal FPR: {val_metrics['normal_road_fpr'] * 100:.1f}%"
                )

            if no_improve_epochs >= patience:
                print(f" -> Early stopping triggered at epoch {epoch}")
                break

        print(f"[*] Fold {fold + 1} Best Macro F1: {best_val_f1:.4f}")
        fold_metrics.append(best_val_metrics)

    # Cross-validation summary
    avg_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
    avg_prec = np.mean([m["macro_precision"] for m in fold_metrics])
    avg_rec = np.mean([m["macro_recall"] for m in fold_metrics])
    avg_fpr = np.mean([m["normal_road_fpr"] for m in fold_metrics])

    print("\n" + "=" * 65)
    print("      HYPERGRID CfC CROSS-VALIDATION FINAL SUMMARY")
    print("=" * 65)
    print(f" -> Mean Macro F1-Score       : {avg_f1:.4f}")
    print(f" -> Mean Macro Precision      : {avg_prec:.4f}")
    print(f" -> Mean Macro Recall         : {avg_rec:.4f}")
    print(f" -> Mean Normal Road FPR      : {avg_fpr * 100:.2f}% (False Alarm Rate)")
    print(f" -> Best Checkpoint Saved at  : {best_model_path}")
    print(f" -> Normalization Scaler      : {scaler_path}")
    print("=" * 65)

    summary = {
        "mean_macro_f1": float(avg_f1),
        "mean_macro_precision": float(avg_prec),
        "mean_macro_recall": float(avg_rec),
        "mean_normal_road_fpr": float(avg_fpr),
        "best_overall_f1": float(best_overall_f1),
        "fold_metrics": fold_metrics,
        "best_model_path": best_model_path,
        "scaler_path": scaler_path,
    }

    summary_file = os.path.join(output_dir, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
# CLI Entrypoint
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train HyperGrid CfC Continuous-Time Deep Learning Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to telemetry dataset directory")
    parser.add_argument("--output_dir", type=str, default="./models/cfc", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--hidden_size", type=int, default=32, help="CfC recurrent units (32-64)")
    parser.add_argument("--seq_len", type=int, default=100, help="Sliding window sequence length")
    parser.add_argument("--stride", type=int, default=30, help="Sliding window stride")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "ce"], help="Loss function")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of StratifiedGroupKFold splits")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'mps', 'auto')")
    parser.add_argument("--no_conv", action="store_true", help="Disable 1D Depthwise Conv frontend")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cfc_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        sequence_length=args.seq_len,
        stride=args.stride,
        loss_type=args.loss,
        k_folds=args.k_folds,
        device_name=args.device,
        use_conv_frontend=not args.no_conv,
    )
