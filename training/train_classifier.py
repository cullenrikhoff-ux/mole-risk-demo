from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

if __name__ == "__main__" and __package__ is None:
    # Allow running as: python training/train_classifier.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import timm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.dataset import ManifestDataset
from training.metrics import accuracy, compute_auc, compute_confusion, sensitivity, specificity

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 3 melanoma classifier training")
    parser.add_argument("--train_csv", default="manifests/train.csv")
    parser.add_argument("--val_csv", default="manifests/val.csv")
    parser.add_argument("--test_csv", default="manifests/test.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model_name", default="efficientnet_b0")
    parser.add_argument("--out_dir", default="results/day3")
    parser.add_argument("--model_out", default="models/classifier_v1.pt")
    return parser.parse_args()


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def compute_pos_weight(csv_path: str) -> Tuple[float, int, int]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Training manifest missing 'label' column.")
    labels = pd.to_numeric(df["label"], errors="raise").astype(int)
    pos_count = int(labels.sum())
    neg_count = int((labels == 0).sum())
    pos_weight = neg_count / max(pos_count, 1)
    return pos_weight, pos_count, neg_count


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, List[int], List[float]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true: List[int] = []
    y_score: List[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1)

        with torch.set_grad_enabled(is_train):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.detach().cpu().int().tolist())
        y_score.extend(torch.sigmoid(logits).detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, y_true, y_score


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, out_path: str) -> None:
    matrix = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix (threshold=0.5)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    for path in (args.train_csv, args.val_csv, args.test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing manifest: {path}")

    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_out)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pos_weight, pos_count, neg_count = compute_pos_weight(args.train_csv)
    print(f"Train labels: pos={pos_count} neg={neg_count} pos_weight={pos_weight:.3f}")

    train_transform, eval_transform = build_transforms(args.img_size)
    train_ds = ManifestDataset(
        args.train_csv, transform=train_transform, img_size=args.img_size
    )
    val_ds = ManifestDataset(args.val_csv, transform=eval_transform, img_size=args.img_size)
    test_ds = ManifestDataset(args.test_csv, transform=eval_transform, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = timm.create_model(args.model_name, pretrained=True, num_classes=1)
    model = model.to(device)

    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_true, val_score = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        val_auc = compute_auc(val_true, val_score)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "best_val_auc": best_val_auc,
                    "epoch": best_epoch,
                },
                args.model_out,
            )
            print(f"Saved best model to {args.model_out}")

    if best_epoch == -1:
        # Fall back to saving the final model if AUC was undefined.
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": args.model_name,
                "img_size": args.img_size,
                "best_val_auc": best_val_auc,
                "epoch": args.epochs,
            },
            args.model_out,
        )
        print(f"Saved final model to {args.model_out}")

    if os.path.exists(args.model_out):
        checkpoint = torch.load(args.model_out, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

    test_loss, test_true, test_score = run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )
    test_auc = compute_auc(test_true, test_score)
    test_pred = [1 if score >= 0.5 else 0 for score in test_score]
    tn, fp, fn, tp = compute_confusion(test_true, test_pred)
    test_sens = sensitivity(tn, fp, fn, tp)
    test_spec = specificity(tn, fp, fn, tp)
    test_acc = accuracy(tn, fp, fn, tp)

    metrics = {
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "sensitivity": test_sens,
        "specificity": test_spec,
        "accuracy": test_acc,
    }

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plot_confusion_matrix(tn, fp, fn, tp, cm_path)

    print(
        "Test metrics: "
        f"auc={test_auc:.4f} acc={test_acc:.4f} sens={test_sens:.4f} spec={test_spec:.4f}"
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
