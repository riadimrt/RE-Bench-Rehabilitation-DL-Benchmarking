"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODULE 6: CONVNEXT + EFFICIENTNETV2 — CNN TERBARU                          ║
║  Novelty: CNN generasi terbaru (2022) untuk rehabilitasi                    ║
║  Dibandingkan dengan CNN di Shalaby et al. (HRNet/ResNet-based)             ║
║                                                                              ║
║  ConvNeXt-Tiny  (2022): "A ConvNet for the 2020s" — Liu et al.              ║
║  EfficientNetV2 (2021): Improved EfficientNet — Tan & Le                    ║
║                                                                              ║
║  Usage:                                                                      ║
║    pip install timm thop                                                     ║
║    python 06_convnext_new_cnn.py --data_dir ./data --output_dir ./output    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (sama persis dengan FC-1~FC-7 RE-Bench)
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 7
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED        = 42   # FC-7: Fixed random seed
N_FOLDS     = 5    # FC-2: Stratified 5-Fold CV
BATCH_SIZE  = 16   # FC-4: Controlled training config

CLASS_NAMES = ['NoObj','Unkn','Ext','Flex','Abd','HExt','Add']

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"  Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS (FC-3: Unified Preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ConvNeXt butuh input 224×224
TRAIN_TF_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET (sama dengan Module 1)
# ─────────────────────────────────────────────────────────────────────────────
class SubsetWithTransform(Dataset):
    def __init__(self, samples, indices, transform):
        self.samples   = samples
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.samples[self.indices[i]]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

def load_dataset(data_dir):
    samples = []
    exts    = {'.bmp','.jpg','.jpeg','.png'}
    for cls_id in range(NUM_CLASSES):
        folder = Path(data_dir) / f'class_{cls_id}'
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in exts:
                samples.append((str(f), cls_id))
    print(f"  Loaded {len(samples)} samples from {data_dir}")
    return samples

def compute_class_weights(labels):
    from collections import Counter
    counter = Counter(labels)
    total   = sum(counter.values())
    weights = [total / (NUM_CLASSES * counter.get(i, 1))
               for i in range(NUM_CLASSES)]
    return torch.FloatTensor(weights)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS — CNN TERBARU
# ─────────────────────────────────────────────────────────────────────────────
def build_convnext_tiny(num_classes=NUM_CLASSES, pretrained=True):
    """
    ConvNeXt-Tiny (2022) — "A ConvNet for the 2020s"
    Liu et al., CVPR 2022
    Params: ~28M, FLOPs: ~4.5G
    Menggabungkan desain modern ViT ke dalam arsitektur CNN murni
    """
    try:
        import timm
        model = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=num_classes
        )
        print(f"  ✓ ConvNeXt-Tiny loaded "
              f"({'pretrained' if pretrained else 'random'})")
        return model
    except ImportError:
        print("  [ERROR] timm not installed: pip install timm")
        sys.exit(1)

def build_efficientnetv2_s(num_classes=NUM_CLASSES, pretrained=True):
    """
    EfficientNetV2-S (2021) — Improved Training-Aware NAS
    Tan & Le, ICML 2021
    Params: ~21M, FLOPs: ~8.8G
    Lebih efisien training dibanding EfficientNet-B0
    """
    try:
        import timm
        model = timm.create_model(
            'efficientnetv2_s',
            pretrained=pretrained,
            num_classes=num_classes
        )
        print(f"  ✓ EfficientNetV2-S loaded "
              f"({'pretrained' if pretrained else 'random'})")
        return model
    except ImportError:
        print("  [ERROR] timm not installed: pip install timm")
        sys.exit(1)

def build_mobilenetv3_large(num_classes=NUM_CLASSES, pretrained=True):
    """
    MobileNetV3-Large (2019) — Searching for MobileNetV3
    Howard et al., ICCV 2019
    Lebih baru dari MobileNetV2 yang dipakai Artikel 2 saat ini
    """
    from torchvision import models
    model = models.mobilenet_v3_large(
        weights='IMAGENET1K_V1' if pretrained else None)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, num_classes)
    print(f"  ✓ MobileNetV3-Large loaded")
    return model

NEW_CNN_MODELS = {
    'ConvNeXt-Tiny':    (build_convnext_tiny,       TRAIN_TF_224, VAL_TF_224),
    'EfficientNetV2-S': (build_efficientnetv2_s,    TRAIN_TF_224, VAL_TF_224),
    'MobileNetV3-Large':(build_mobilenetv3_large,   TRAIN_TF,     VAL_TF),
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (FC-4: Controlled training configuration)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds_all, true_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        p    = model(imgs).argmax(1).cpu().numpy()
        preds_all.extend(p)
        true_all.extend(labels.numpy())
    return np.array(preds_all), np.array(true_all)

def train_fold(model_name, builder, train_tf, val_tf,
               samples, train_idx, val_idx,
               epochs_warmup=5, epochs_finetune=25):
    """
    Two-phase training (sama dengan Module 1 — FC-4 compliance):
    Phase 1: Warmup — freeze backbone
    Phase 2: Fine-tune — unfreeze all
    """
    train_labels = [samples[i][1] for i in train_idx]
    cw = compute_class_weights(train_labels).to(DEVICE)

    tr_ds  = SubsetWithTransform(samples, train_idx, train_tf)
    val_ds = SubsetWithTransform(samples, val_idx,   val_tf)
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)

    model     = builder(NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Phase 1: Warmup (freeze backbone)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze head only
    try:
        for param in model.head.parameters():
            param.requires_grad = True
    except AttributeError:
        try:
            for param in model.classifier.parameters():
                param.requires_grad = True
        except AttributeError:
            for param in model.parameters():
                param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs_warmup):
        train_one_epoch(model, tr_dl, opt, criterion)

    # Phase 2: Fine-tune (unfreeze all)
    for param in model.parameters():
        param.requires_grad = True
    opt  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_finetune)

    best_acc, best_state = 0.0, None
    for _ in range(epochs_finetune):
        train_one_epoch(model, tr_dl, opt, criterion)
        sch.step()
        preds, true = evaluate(model, val_dl)
        acc = accuracy_score(true, preds)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return evaluate(model, val_dl)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_new_cnn_evaluation(data_dir, output_dir,
                            epochs_warmup=5, epochs_finetune=25,
                            models_to_run=None):
    print("\n" + "="*78)
    print("  CNN TERBARU — STRATIFIED 5-FOLD CV")
    print("  ConvNeXt-Tiny | EfficientNetV2-S | MobileNetV3-Large")
    print("  Novelty: CNN generasi 2021-2022 vs Shalaby (2019-era CNN)")
    print("="*78)

    os.makedirs(output_dir, exist_ok=True)
    samples    = load_dataset(data_dir)
    all_labels = np.array([s[1] for s in samples])

    print(f"\n  Total samples: {len(all_labels)}")
    print(f"  Device: {DEVICE}")

    if models_to_run is None:
        models_to_run = list(NEW_CNN_MODELS.keys())

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_res = []

    for model_name in models_to_run:
        if model_name not in NEW_CNN_MODELS:
            print(f"  ⚠ Unknown model: {model_name}")
            continue

        builder, tr_tf, val_tf = NEW_CNN_MODELS[model_name]

        print(f"\n{'─'*78}")
        print(f"  MODEL: {model_name}")
        print(f"{'─'*78}")

        fold_accs, fold_f1s = [], []
        all_preds, all_true = [], []
        t0 = time.time()

        for fold, (tr_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(all_labels)), all_labels)):
            print(f"  Fold {fold+1}/{N_FOLDS} | "
                  f"Train:{len(tr_idx)} Val:{len(val_idx)}", end=' ')
            sys.stdout.flush()

            preds, true = train_fold(
                model_name, builder, tr_tf, val_tf,
                samples, tr_idx, val_idx,
                epochs_warmup, epochs_finetune)

            acc = accuracy_score(true, preds) * 100
            f1  = f1_score(true, preds, average='macro', zero_division=0) * 100
            fold_accs.append(acc)
            fold_f1s.append(f1)
            all_preds.extend(preds)
            all_true.extend(true)
            print(f"→ Acc:{acc:.2f}% F1:{f1:.2f}%")

        elapsed = time.time() - t0

        # Aggregate
        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        mean_f1  = np.mean(fold_f1s)
        prec = precision_score(all_true, all_preds,
                               average='macro', zero_division=0) * 100
        rec  = recall_score(all_true, all_preds,
                            average='macro', zero_division=0) * 100
        cm   = confusion_matrix(all_true, all_preds,
                                labels=list(range(NUM_CLASSES)))

        print(f"\n  ─── {model_name} RESULTS (S5-CV) ───")
        print(f"  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  F1 Macro : {mean_f1:.2f}%")
        print(f"  Precision: {prec:.2f}%  Recall: {rec:.2f}%")
        print(f"  Time     : {elapsed:.1f}s")

        # Per-class
        print("\n  Per-Class Report:")
        print(classification_report(all_true, all_preds,
              target_names=CLASS_NAMES, zero_division=0))

        # Confusion matrix
        print("  Confusion Matrix:")
        header = f"  {'':20}" + "".join(f"{c:>8}" for c in CLASS_NAMES)
        print(header)
        for i, row in enumerate(cm):
            rc = cm[i,i]/max(cm[i].sum(),1)*100
            print(f"  {CLASS_NAMES[i]:<20}"
                  + "".join(f"{v:8d}" for v in row)
                  + f"  ← Recall={rc:.1f}%")

        all_res.append({
            'Model':           model_name,
            'Paradigm':        'CNN',
            'Generation':      '2021-2022',
            'Accuracy_Mean%':  f"{mean_acc:.4f}",
            'Accuracy_Std%':   f"{std_acc:.4f}",
            'F1_Score%':       f"{mean_f1:.4f}",
            'Precision%':      f"{prec:.4f}",
            'Recall%':         f"{rec:.4f}",
            'S5CV_Time_s':     f"{elapsed:.1f}",
            'CPU_Native':      'No',
            'GPU_Required':    'Yes',
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*78)
    print("  SUMMARY — NEW CNN ARCHITECTURES vs EXISTING MODELS")
    print("="*78)

    existing = {
        'MobileNetV2 (existing)':    (97.22, 1.12, 96.00, 2187.7),
        'EfficientNet-B0 (existing)':(96.87, 1.49, 95.54, 3057.7),
        'ViT-Tiny (existing)':       (97.22, 1.18, 96.73, 6335.7),
    }

    print(f"\n  {'Model':<25} {'Acc(%)':>10} {'±Std':>8} {'F1(%)':>8} "
          f"{'Time(s)':>10} {'Gen':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")

    for name, (acc, std, f1, t) in existing.items():
        print(f"  {name:<25} {acc:>10.2f} {std:>8.2f} {f1:>8.2f} "
              f"{t:>10.1f} {'2018-19':>8}")

    for r in all_res:
        print(f"  {r['Model']:<25} "
              f"{float(r['Accuracy_Mean%']):>10.2f} "
              f"{float(r['Accuracy_Std%']):>8.2f} "
              f"{float(r['F1_Score%']):>8.2f} "
              f"{float(r['S5CV_Time_s']):>10.1f} "
              f"{'2021-22':>8}")

    # Save CSV
    df = pd.DataFrame(all_res)
    csv_path = os.path.join(output_dir, 'hasil_new_cnn_cv.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Results saved → {csv_path}")

    return all_res


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='New CNN Architectures (2021-2022) for Novelty')
    parser.add_argument('--data_dir',        type=str, default='./data')
    parser.add_argument('--output_dir',      type=str, default='./output')
    parser.add_argument('--epochs_warmup',   type=int, default=5)
    parser.add_argument('--epochs_finetune', type=int, default=25)
    parser.add_argument('--models', nargs='+',
                        default=['ConvNeXt-Tiny',
                                 'EfficientNetV2-S',
                                 'MobileNetV3-Large'],
                        choices=['ConvNeXt-Tiny',
                                 'EfficientNetV2-S',
                                 'MobileNetV3-Large'])
    args = parser.parse_args()

    run_new_cnn_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs_warmup=args.epochs_warmup,
        epochs_finetune=args.epochs_finetune,
        models_to_run=args.models,
    )
