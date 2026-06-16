"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODULE 1: CNN TRANSFER LEARNING                                             ║
║  MobileNetV2 + EfficientNet-B0 + ResNet18                                   ║
║  Evaluated with Stratified 5-Fold Cross-Validation (S5-CV)                  ║
║  Compatible with: Paper "Comparative Evaluation..."                          ║
║                                                                              ║
║  Input : RGB frames (BMP files), same directory as classical baseline        ║
║  Output: hasil_cnn_cv.csv, confusion matrices, per-class report             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Requirements:
    pip install torch torchvision tqdm scikit-learn pillow numpy pandas

Usage:
    python 01_cnn_transfer_learning.py --data_dir PATH_TO_FRAMES --epochs 30
    python 01_cnn_transfer_learning.py --data_dir . --epochs 30 --models mobilenet efficientnet resnet18

Architecture Notes:
    - MobileNetV2 : ~3.4M params, ImageNet pretrained, classifier head replaced
    - EfficientNet-B0: ~5.3M params, compound scaling, best accuracy/param ratio
    - ResNet18    : ~11.7M params, residual connections, strong baseline
    - All: freeze backbone → train head (5 epochs) → unfreeze all → fine-tune (25 epochs)
    - Data augmentation: random crop, horizontal flip, color jitter, rotation
"""

import os
import sys
import csv
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Tidak Ada Objek',   # 0
    'Tdk Dikenal',       # 1
    'Ekstensi',          # 2
    'Fleksi',            # 3
    'Abduksi',           # 4
    'Hiperekstensi',     # 5
    'Adduksi',           # 6
]
CLASS_SHORT = ['NoObj','Unkn','Ext','Flex','Abd','HExt','Add']
NUM_CLASSES = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
# Augmentation for training
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MotionDataset(Dataset):
    """
    Expects BMP files named with class label encoded in filename or folder.
    Supported structures:
        a) Flat directory: files named like "class2_frame001.bmp"
        b) Subfolder per class: data_dir/class_2/frame001.bmp
        c) Loads ciri_chain_code.mat label column as ground truth (fallback)
    """
    def __init__(self, data_dir, transform=None, labels_csv=None):
        self.transform = transform
        self.samples = []   # (path, label)

        data_dir = Path(data_dir)

        # ── Try subfolder structure first ──────────────────────────────────
        subfolders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        if subfolders:
            for folder in subfolders:
                # Extract class index from folder name (e.g. "class_2", "2", "Ekstensi")
                label = self._parse_folder_label(folder.name)
                if label is not None:
                    for img_path in sorted(folder.glob('*.bmp')):
                        self.samples.append((str(img_path), label))
                    for img_path in sorted(folder.glob('*.jpg')):
                        self.samples.append((str(img_path), label))
                    for img_path in sorted(folder.glob('*.png')):
                        self.samples.append((str(img_path), label))
            if self.samples:
                print(f"  [Dataset] Loaded {len(self.samples)} samples from {len(subfolders)} class folders")
                return

        # ── Flat directory: parse label from filename ──────────────────────
        all_imgs = sorted(list(data_dir.glob('*.bmp')) +
                          list(data_dir.glob('*.jpg')) +
                          list(data_dir.glob('*.png')))
        for img_path in all_imgs:
            label = self._parse_filename_label(img_path.name)
            if label is not None:
                self.samples.append((str(img_path), label))

        # ── Fallback: load from CSV if provided ───────────────────────────
        if not self.samples and labels_csv:
            df = pd.read_csv(labels_csv)
            # Assumes columns: 'filepath', 'label'
            for _, row in df.iterrows():
                self.samples.append((str(data_dir / row['filepath']), int(row['label'])))
            print(f"  [Dataset] Loaded {len(self.samples)} samples from CSV")
            return

        if self.samples:
            print(f"  [Dataset] Loaded {len(self.samples)} samples from flat dir")
        else:
            raise ValueError(
                f"No samples found in {data_dir}.\n"
                "Please organize images into subfolders named 'class_0' through 'class_6',\n"
                "or provide a labels_csv file with columns 'filepath' and 'label'."
            )

    def _parse_folder_label(self, name):
        """Parse class label from folder name."""
        name = name.lower().strip()
        mapping = {
            '0': 0, 'class_0': 0, 'class0': 0, 'tidakada': 0, 'tidak_ada': 0, 'no_object': 0,
            '1': 1, 'class_1': 1, 'class1': 1, 'unknown': 1, 'tdkdikenal': 1,
            '2': 2, 'class_2': 2, 'class2': 2, 'ekstensi': 2, 'extension': 2,
            '3': 3, 'class_3': 3, 'class3': 3, 'fleksi': 3, 'flexion': 3,
            '4': 4, 'class_4': 4, 'class4': 4, 'abduksi': 4, 'abduction': 4,
            '5': 5, 'class_5': 5, 'class5': 5, 'hiperekstensi': 5, 'hyperextension': 5,
            '6': 6, 'class_6': 6, 'class6': 6, 'adduksi': 6, 'adduction': 6,
        }
        return mapping.get(name, None)

    def _parse_filename_label(self, name):
        """Parse label from filename like 'class2_001.bmp' or 'ekstensi_001.bmp'."""
        name_lower = name.lower()
        for key, val in [('class0',0),('class1',1),('class2',2),('class3',3),
                         ('class4',4),('class5',5),('class6',6),
                         ('ekstensi',2),('fleksi',3),('abduksi',4),
                         ('hiperekstensi',5),('adduksi',6)]:
            if key in name_lower:
                return val
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenetv2(num_classes=NUM_CLASSES, pretrained=True):
    """MobileNetV2: lightweight, 3.4M params, excellent for limited data."""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    """EfficientNet-B0: best accuracy-per-parameter, compound scaling."""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def build_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    """ResNet18: deeper residual connections, strong baseline."""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


MODEL_BUILDERS = {
    'mobilenet':   build_mobilenetv2,
    'efficientnet': build_efficientnet_b0,
    'resnet18':    build_resnet18,
}


def freeze_backbone(model, model_name):
    """Freeze all layers except classifier head for initial warm-up."""
    if model_name == 'resnet18':
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    elif model_name == 'mobilenet':
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    elif model_name == 'efficientnet':
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced dataset."""
    counter = Counter(labels)
    total = sum(counter.values())
    weights = [total / (NUM_CLASSES * counter.get(i, 1)) for i in range(NUM_CLASSES)]
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_model_on_fold(model_name, train_idx, val_idx, full_dataset,
                        epochs_warmup=5, epochs_finetune=25,
                        lr_warmup=1e-3, lr_finetune=1e-4, batch_size=16):
    """
    Two-phase training:
    Phase 1 (warm-up): freeze backbone, train head only with lr=1e-3
    Phase 2 (fine-tune): unfreeze all, train with lr=1e-4 and cosine LR
    """
    # Build datasets with appropriate transforms
    class SubsetWithTransform(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            img_path, label = self.dataset.samples[self.indices[i]]
            img = Image.open(img_path).convert('RGB')
            return self.transform(img), label

    train_ds = SubsetWithTransform(full_dataset, train_idx, TRAIN_TRANSFORM)
    val_ds   = SubsetWithTransform(full_dataset, val_idx,   VAL_TRANSFORM)

    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    class_weights = compute_class_weights(train_labels).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    model = MODEL_BUILDERS[model_name](num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Phase 1: Warm-up (frozen backbone) ──────────────────────────────
    freeze_backbone(model, model_name)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr_warmup, weight_decay=1e-4)

    best_val_acc, best_state = 0.0, None
    for epoch in range(epochs_warmup):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

    # ── Phase 2: Full fine-tuning ────────────────────────────────────────
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_finetune)

    for epoch in range(epochs_finetune):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        # Track best model on validation
        val_preds, val_labels = evaluate(model, val_loader, DEVICE)
        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best state
    model.load_state_dict(best_state)
    val_preds, val_labels = evaluate(model, val_loader, DEVICE)

    return val_preds, val_labels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(data_dir, model_names, n_splits=5,
                   epochs_warmup=5, epochs_finetune=25, output_dir='.'):
    print("\n" + "="*78)
    print("  CNN TRANSFER LEARNING — STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("="*78)
    print(f"  Data dir : {data_dir}")
    print(f"  Models   : {model_names}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {epochs_warmup} warmup + {epochs_finetune} finetune per fold")
    print("="*78)

    # Load dataset (no transform at dataset level; applied per fold)
    full_dataset = MotionDataset(data_dir)
    all_labels = np.array(full_dataset.get_labels())

    print(f"\n  Total samples: {len(all_labels)}")
    print("  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        n = (all_labels == i).sum()
        print(f"    Class {i} ({name:20s}): {n:4d} ({100*n/len(all_labels):.1f}%)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []
    cv_summary = []

    for model_name in model_names:
        print(f"\n{'─'*78}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'─'*78}")

        fold_accs, fold_f1s = [], []
        all_fold_preds, all_fold_labels = [], []

        t0 = time.time()
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            print(f"  Fold {fold+1}/{n_splits}  |  Train: {len(train_idx)} | Val: {len(val_idx)}", end=' ')
            sys.stdout.flush()

            val_preds, val_labels = train_model_on_fold(
                model_name, train_idx, val_idx, full_dataset,
                epochs_warmup=epochs_warmup, epochs_finetune=epochs_finetune
            )

            acc = accuracy_score(val_labels, val_preds) * 100
            f1  = f1_score(val_labels, val_preds, average='macro', zero_division=0) * 100
            fold_accs.append(acc)
            fold_f1s.append(f1)
            all_fold_preds.extend(val_preds)
            all_fold_labels.extend(val_labels)

            print(f"→ Acc: {acc:.2f}%  F1: {f1:.2f}%")

        elapsed = time.time() - t0

        # Aggregate metrics
        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        mean_f1  = np.mean(fold_f1s)

        all_fold_preds  = np.array(all_fold_preds)
        all_fold_labels = np.array(all_fold_labels)

        prec = precision_score(all_fold_labels, all_fold_preds, average='macro', zero_division=0) * 100
        rec  = recall_score(all_fold_labels, all_fold_preds, average='macro', zero_division=0) * 100
        cm   = confusion_matrix(all_fold_labels, all_fold_preds, labels=list(range(NUM_CLASSES)))

        print(f"\n  {'─'*60}")
        print(f"  {model_name.upper()} RESULTS (S5-CV)")
        print(f"  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  F1 Macro : {mean_f1:.2f}%")
        print(f"  Precision: {prec:.2f}%  Recall: {rec:.2f}%")
        print(f"  Time     : {elapsed:.1f}s")
        print(f"  {'─'*60}")

        # Per-class report
        print("\n  Per-Class Report:")
        report = classification_report(
            all_fold_labels, all_fold_preds,
            target_names=CLASS_SHORT, zero_division=0
        )
        for line in report.split('\n'):
            print(f"    {line}")

        # Confusion matrix display
        print("\n  Confusion Matrix:")
        header = f"  {'':20s}" + "".join(f"{c:>8s}" for c in CLASS_SHORT)
        print(header)
        for i, row in enumerate(cm):
            recalls = row[i] / max(row.sum(), 1) * 100
            print(f"  {CLASS_SHORT[i]:20s}" + "".join(f"{v:8d}" for v in row) +
                  f"  ← Recall={recalls:.1f}%")

        all_results.append({
            'model': model_name,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_f1': mean_f1,
            'precision': prec,
            'recall': rec,
            'time_sec': elapsed,
            'confusion_matrix': cm.tolist(),
        })

        cv_summary.append({
            'Model': model_name,
            'Accuracy_Mean(%)': f"{mean_acc:.4f}",
            'Accuracy_Std(%)': f"{std_acc:.4f}",
            'F1_Score(%)': f"{mean_f1:.4f}",
            'Precision(%)': f"{prec:.4f}",
            'Recall(%)': f"{rec:.4f}",
            'Train_Time(s)': f"{elapsed:.1f}",
        })

    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n" + "="*78)
    print("  FINAL COMPARISON — CNN TRANSFER LEARNING (S5-CV)")
    print("="*78)
    print(f"  {'Model':20s} {'Acc (%)':>10s} {'Std':>8s} {'F1 (%)':>8s} {'Prec':>8s} {'Rec':>8s} {'Time':>8s}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in all_results:
        star = ' ★' if r == max(all_results, key=lambda x: x['mean_acc']) else ''
        print(f"  {r['model']:20s} {r['mean_acc']:>10.2f} "
              f"{r['std_acc']:>8.2f} {r['mean_f1']:>8.2f} "
              f"{r['precision']:>8.2f} {r['recall']:>8.2f} "
              f"{r['time_sec']:>7.1f}s{star}")

    # ── Save CSV ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'hasil_cnn_cv.csv')
    df = pd.DataFrame(cv_summary)
    df.to_csv(out_csv, index=False)
    print(f"\n  ✓ Results saved → {out_csv}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Transfer Learning for Motion Classification')
    parser.add_argument('--data_dir',  type=str, default='.',
                        help='Directory containing image subfolders or flat images')
    parser.add_argument('--models',    nargs='+',
                        default=['mobilenet', 'efficientnet', 'resnet18'],
                        choices=['mobilenet', 'efficientnet', 'resnet18'],
                        help='Models to evaluate')
    parser.add_argument('--epochs_warmup',   type=int, default=5)
    parser.add_argument('--epochs_finetune', type=int, default=25)
    parser.add_argument('--output_dir',      type=str, default='./output')
    parser.add_argument('--batch_size',      type=int, default=16)
    args = parser.parse_args()

    results = run_evaluation(
        data_dir=args.data_dir,
        model_names=args.models,
        epochs_warmup=args.epochs_warmup,
        epochs_finetune=args.epochs_finetune,
        output_dir=args.output_dir,
    )
