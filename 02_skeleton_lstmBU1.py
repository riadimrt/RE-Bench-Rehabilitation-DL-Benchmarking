"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODULE 2: MEDIAPIPE SKELETON + LSTM / BiLSTM / Attention-LSTM             ║
║  Pose keypoint extraction → temporal sequence modeling                      ║
║  Evaluated with Stratified 5-Fold Cross-Validation (S5-CV)                  ║
║                                                                              ║
║  Input : RGB frames (or BMP/PNG) in class-organized folders                 ║
║  Output: hasil_skeleton_lstm_cv.csv, per-class report                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Requirements:
    pip install mediapipe torch scikit-learn pandas numpy tqdm pillow opencv-python

Architecture:
    1. MediaPipe Pose: extract 33 keypoints × (x, y, z, visibility) = 132-dim vector/frame
    2. Frame sequences of length SEQ_LEN constructed via sliding window
    3. Models:
        a) LSTM     : single-direction LSTM, 2 layers, hidden=256
        b) BiLSTM   : bidirectional LSTM, captures past+future context
        c) Attn-LSTM: LSTM + scaled dot-product attention over time steps

Notes:
    - Single-frame mode: SEQ_LEN=1, effectively MLP on skeleton features
    - For video sequences: SEQ_LEN >= 8 recommended
    - Falls back to HOG skeleton simulation if MediaPipe fails on a frame
"""

import os
import sys
import csv
import time
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['NoObj','Unkn','Ext','Flex','Abd','HExt','Add']
NUM_CLASSES  = 7
SEQ_LEN      = 1        # frames per sequence; use 1 for static frame-based
FEAT_DIM     = 132      # 33 keypoints × 4 (x, y, z, visibility)
HIDDEN_DIM   = 256
N_LAYERS     = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"  Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def init_mediapipe():
    """Initialize MediaPipe Pose with optimal settings."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,         # 0=lite, 1=full, 2=heavy
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        print("  ✓ MediaPipe Pose initialized")
        return pose, mp_pose
    except ImportError:
        print("  ⚠ MediaPipe not available — using fallback feature extractor")
        return None, None


def extract_mediapipe_features(img_path, pose_model):
    """
    Extract 132-dim skeleton feature from image.
    Returns zeros if no person detected.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.array(Image.open(str(img_path)).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)

    if results.pose_landmarks:
        kpts = []
        for lm in results.pose_landmarks.landmark:
            kpts.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(kpts, dtype=np.float32)   # (132,)
    else:
        return np.zeros(FEAT_DIM, dtype=np.float32)


def extract_hog_skeleton_fallback(img_path, feat_dim=132):
    """
    Fallback feature when MediaPipe fails:
    Uses HOG + silhouette moments to produce a pose-like descriptor.
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(feat_dim, dtype=np.float32)

        img = cv2.resize(img, (64, 64))

        # HOG-like features (8×8 cells, 9 orientations = 576-dim → PCA to feat_dim)
        # Simplified: use pixel patch statistics
        patches = img.reshape(8, 8, 8, 8)
        feats = np.concatenate([
            patches.mean(axis=(2,3)).flatten(),        # 64 means
            patches.std(axis=(2,3)).flatten(),         # 64 stds
        ])
        # Pad or trim to feat_dim
        if len(feats) >= feat_dim:
            return feats[:feat_dim].astype(np.float32)
        else:
            return np.pad(feats, (0, feat_dim - len(feats))).astype(np.float32)
    except Exception:
        return np.zeros(feat_dim, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    """Precomputes skeleton features from all frames; stores in memory."""

    def __init__(self, data_dir, pose_model, seq_len=SEQ_LEN, augment=False):
        self.seq_len  = seq_len
        self.augment  = augment
        self.samples  = []   # (feature_seq, label)

        data_dir = Path(data_dir)
        all_frames = []  # (path, label)

        # Load frames organized in subfolders
        label_map = {
            '0':0,'class_0':0,'class0':0,'tidakada':0,'no_object':0,
            '1':1,'class_1':1,'class1':1,'unknown':1,
            '2':2,'class_2':2,'class2':2,'ekstensi':2,'extension':2,
            '3':3,'class_3':3,'class3':3,'fleksi':3,'flexion':3,
            '4':4,'class_4':4,'class4':4,'abduksi':4,'abduction':4,
            '5':5,'class_5':5,'class5':5,'hiperekstensi':5,'hyperextension':5,
            '6':6,'class_6':6,'class6':6,'adduksi':6,'adduction':6,
        }

        subfolders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        if subfolders:
            for folder in subfolders:
                label = label_map.get(folder.name.lower())
                if label is not None:
                    imgs = sorted(list(folder.glob('*.bmp')) +
                                  list(folder.glob('*.jpg')) +
                                  list(folder.glob('*.png')))
                    for p in imgs:
                        all_frames.append((p, label))
        else:
            # Flat directory
            for img_path in sorted(data_dir.glob('*.bmp')):
                for key, val in label_map.items():
                    if key in img_path.name.lower():
                        all_frames.append((img_path, val))
                        break

        if not all_frames:
            raise ValueError(f"No frames found in {data_dir}")

        print(f"  Extracting skeleton features from {len(all_frames)} frames...")
        t0 = time.time()

        # Precompute features
        frame_features = []
        frame_labels   = []
        for i, (img_path, label) in enumerate(all_frames):
            if pose_model is not None:
                feat = extract_mediapipe_features(img_path, pose_model)
            else:
                feat = extract_hog_skeleton_fallback(img_path)

            frame_features.append(feat)
            frame_labels.append(label)

            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(all_frames)} frames processed...", end='\r')

        print(f"\n  ✓ Feature extraction done ({time.time()-t0:.1f}s)")

        # Build sequences (sliding window) or single-frame
        frame_features = np.array(frame_features)
        frame_labels   = np.array(frame_labels)

        if seq_len == 1:
            # Single-frame mode
            for i in range(len(frame_features)):
                self.samples.append((
                    frame_features[i:i+1],   # (1, 132)
                    frame_labels[i]
                ))
        else:
            # Sliding window: use frames of same class if sequential ordering unknown
            for cls in range(NUM_CLASSES):
                cls_idx = np.where(frame_labels == cls)[0]
                cls_feats = frame_features[cls_idx]
                for start in range(0, max(1, len(cls_feats) - seq_len + 1)):
                    end = start + seq_len
                    if end > len(cls_feats):
                        # Pad with zeros
                        seq = np.pad(cls_feats[start:],
                                     ((0, end - len(cls_feats)), (0, 0)))
                    else:
                        seq = cls_feats[start:end]
                    self.samples.append((seq, cls))   # (seq_len, 132)

        print(f"  Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_seq, label = self.samples[idx]
        feat_tensor = torch.FloatTensor(feat_seq)  # (seq_len, 132)

        if self.augment:
            # Gaussian noise augmentation
            noise = torch.randn_like(feat_tensor) * 0.01
            feat_tensor = feat_tensor + noise

        return feat_tensor, label

    def get_labels(self):
        return [s[1] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Standard LSTM for sequential skeleton features."""
    def __init__(self, input_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq, feat)
        B, S, F = x.shape
        # Normalize per feature
        x_flat = x.reshape(B*S, F)
        x_flat = self.input_norm(x_flat)
        x = x_flat.reshape(B, S, F)

        out, (hn, _) = self.lstm(x)
        # Use last hidden state
        last = hn[-1]  # (batch, hidden)
        return self.classifier(last)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM — captures both temporal directions."""
    def __init__(self, input_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),   # ×2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, S, F = x.shape
        x_flat = x.reshape(B*S, F)
        x_flat = self.input_norm(x_flat)
        x = x_flat.reshape(B, S, F)

        out, (hn, _) = self.lstm(x)
        # Concatenate last hidden from both directions
        last = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, hidden*2)
        return self.classifier(last)


class AttentionLSTM(nn.Module):
    """LSTM + Scaled Dot-Product Attention over time steps."""
    def __init__(self, input_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.input_norm  = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key   = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, S, F = x.shape
        x_flat = x.reshape(B*S, F)
        x_flat = self.input_norm(x_flat)
        x = x_flat.reshape(B, S, F)

        out, _ = self.lstm(x)   # out: (batch, seq, hidden)

        # Scaled dot-product self-attention over sequence
        Q = self.attn_query(out)
        K = self.attn_key(out)
        V = self.attn_value(out)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, S, S)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, V)   # (B, S, hidden)

        # Global average pooling over time
        pooled = context.mean(dim=1)   # (B, hidden)
        return self.classifier(pooled)


MODEL_CLASSES = {
    'lstm':      LSTMClassifier,
    'bilstm':    BiLSTMClassifier,
    'attn_lstm': AttentionLSTM,
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        total_loss += loss.item() * len(labels)
    return total_loss / total, 100 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for feats, labels in loader:
        feats = feats.to(device)
        preds = model(feats).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_skeleton_model(model_cls, train_idx, val_idx, full_dataset,
                         epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train and evaluate one fold."""
    from torch.utils.data import Subset

    class AugmentedSubset(Dataset):
        def __init__(self, ds, idx, augment):
            self.ds = ds; self.idx = idx; self.aug = augment
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            feat, label = self.ds[self.idx[i]]
            if self.aug:
                feat = feat + torch.randn_like(feat) * 0.01
            return feat, label

    train_ds = AugmentedSubset(full_dataset, train_idx, augment=True)
    val_ds   = AugmentedSubset(full_dataset, val_idx,   augment=False)

    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    counter = Counter(train_labels)
    total = sum(counter.values())
    weights = torch.FloatTensor([
        total / (NUM_CLASSES * counter.get(i, 1)) for i in range(NUM_CLASSES)
    ]).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_cls().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_acc, best_state = 0.0, None
    patience, no_improve = 15, 0

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        val_preds, val_labels_arr = eval_epoch(model, val_loader, DEVICE)
        val_acc = accuracy_score(val_labels_arr, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    val_preds, val_labels_arr = eval_epoch(model, val_loader, DEVICE)
    return val_preds, val_labels_arr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_skeleton_evaluation(data_dir, model_names, n_splits=5,
                             seq_len=SEQ_LEN, output_dir='.'):
    print("\n" + "="*78)
    print("  MEDIAPIPE SKELETON + LSTM/BiLSTM/Attn-LSTM — S5-CV")
    print("="*78)

    # Init MediaPipe
    pose_model, _ = init_mediapipe()

    # Build dataset
    full_dataset = SkeletonDataset(data_dir, pose_model, seq_len=seq_len)
    all_labels   = np.array(full_dataset.get_labels())

    print(f"\n  Total samples : {len(all_labels)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = []
    csv_rows    = []

    for model_name in model_names:
        print(f"\n{'─'*78}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'─'*78}")

        model_cls = MODEL_CLASSES[model_name]
        fold_accs, fold_f1s = [], []
        all_preds_agg, all_labels_agg = [], []

        t0 = time.time()
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            print(f"  Fold {fold+1}/{n_splits}  Train:{len(train_idx)} Val:{len(val_idx)}", end=' ')
            sys.stdout.flush()

            val_preds, val_labels_fold = train_skeleton_model(
                model_cls, train_idx, val_idx, full_dataset
            )

            acc = accuracy_score(val_labels_fold, val_preds) * 100
            f1  = f1_score(val_labels_fold, val_preds, average='macro', zero_division=0) * 100
            fold_accs.append(acc)
            fold_f1s.append(f1)
            all_preds_agg.extend(val_preds)
            all_labels_agg.extend(val_labels_fold)
            print(f"→ Acc:{acc:.2f}% F1:{f1:.2f}%")

        elapsed = time.time() - t0
        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        mean_f1  = np.mean(fold_f1s)

        all_preds_agg  = np.array(all_preds_agg)
        all_labels_agg = np.array(all_labels_agg)
        prec = precision_score(all_labels_agg, all_preds_agg, average='macro', zero_division=0) * 100
        rec  = recall_score(all_labels_agg, all_preds_agg, average='macro', zero_division=0) * 100
        cm   = confusion_matrix(all_labels_agg, all_preds_agg, labels=list(range(NUM_CLASSES)))

        print(f"\n  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  F1 Macro : {mean_f1:.2f}%   Prec: {prec:.2f}%  Rec: {rec:.2f}%")
        print(f"  Time     : {elapsed:.1f}s")

        print("\n  Per-Class Report:")
        report = classification_report(all_labels_agg, all_preds_agg,
                                       target_names=CLASS_NAMES, zero_division=0)
        for line in report.split('\n'):
            print(f"    {line}")

        print("  Confusion Matrix:")
        header = f"  {'':15s}" + "".join(f"{c:>8s}" for c in CLASS_NAMES)
        print(header)
        for i, row in enumerate(cm):
            rec_i = row[i] / max(row.sum(), 1) * 100
            print(f"  {CLASS_NAMES[i]:15s}" + "".join(f"{v:8d}" for v in row) +
                  f"  ← {rec_i:.1f}%")

        all_results.append({
            'model': model_name,
            'mean_acc': mean_acc, 'std_acc': std_acc,
            'mean_f1': mean_f1, 'precision': prec, 'recall': rec,
            'time_sec': elapsed,
        })
        csv_rows.append({
            'Model': f"Skeleton+{model_name.upper()}",
            'Accuracy_Mean(%)': f"{mean_acc:.4f}",
            'Accuracy_Std(%)': f"{std_acc:.4f}",
            'F1_Score(%)': f"{mean_f1:.4f}",
            'Precision(%)': f"{prec:.4f}",
            'Recall(%)': f"{rec:.4f}",
            'Train_Time(s)': f"{elapsed:.1f}",
        })

    # Summary
    print("\n" + "="*78)
    print("  FINAL SUMMARY — SKELETON + LSTM (S5-CV)")
    print("="*78)
    print(f"  {'Model':15s} {'Acc(%)':>10s} {'Std':>8s} {'F1(%)':>8s} {'Prec':>8s} {'Rec':>8s}")
    for r in all_results:
        star = ' ★' if r == max(all_results, key=lambda x: x['mean_acc']) else ''
        print(f"  {r['model']:15s} {r['mean_acc']:>10.2f} {r['std_acc']:>8.2f} "
              f"{r['mean_f1']:>8.2f} {r['precision']:>8.2f} {r['recall']:>8.2f}{star}")

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'hasil_skeleton_lstm_cv.csv')
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\n  ✓ Saved → {out_csv}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skeleton LSTM for Motion Classification')
    parser.add_argument('--data_dir',   type=str, default='.')
    parser.add_argument('--models',     nargs='+', default=['lstm', 'bilstm', 'attn_lstm'],
                        choices=['lstm', 'bilstm', 'attn_lstm'])
    parser.add_argument('--seq_len',    type=int, default=1,
                        help='Sequence length (1=single frame, 8+=temporal)')
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    run_skeleton_evaluation(
        data_dir=args.data_dir,
        model_names=args.models,
        seq_len=args.seq_len,
        output_dir=args.output_dir,
    )
