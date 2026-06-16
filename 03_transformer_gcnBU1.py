"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODULE 3: VISION TRANSFORMER (ViT) + LIGHTWEIGHT GCN                      ║
║  ViT-tiny (from scratch / pretrained) + Skeleton Graph Convolutional Net    ║
║  Evaluated with Stratified 5-Fold Cross-Validation                          ║
║                                                                              ║
║  ⚠ WARNING: ViT from scratch requires LARGE dataset (>>5K samples).        ║
║    For 863 frames, use ViT with pretrained weights only.                    ║
║    GCN requires MediaPipe skeleton; falls back to heuristic graph.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Requirements:
    pip install torch torchvision scikit-learn pandas numpy pillow
    pip install timm  (for pretrained ViT)
    pip install mediapipe opencv-python  (for GCN)

Models:
    1. ViT-Tiny (pretrained via timm library, fine-tuned)
    2. Lightweight GCN on 33-node skeleton graph
"""

import os
import sys
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
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 7
CLASS_NAMES = ['NoObj','Unkn','Ext','Flex','Abd','HExt','Add']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {DEVICE}")
if not torch.cuda.is_available():
    print("  ⚠ GPU not detected. ViT training will be slow. Consider Google Colab.")


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: VISION TRANSFORMER (ViT)
# ═══════════════════════════════════════════════════════════════════════════════

# ── ViT transforms ──────────────────────────────────────────────────────────
VIT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
VIT_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


class MotionFrameDataset(Dataset):
    """Shared image dataset for ViT."""
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples   = []
        data_dir = Path(data_dir)
        label_map = {
            '0':0,'class_0':0,'class0':0,'tidakada':0,'no_object':0,
            '1':1,'class_1':1,'class1':1,'unknown':1,
            '2':2,'class_2':2,'class2':2,'ekstensi':2,'extension':2,
            '3':3,'class_3':3,'class3':3,'fleksi':3,'flexion':3,
            '4':4,'class_4':4,'class4':4,'abduksi':4,'abduction':4,
            '5':5,'class_5':5,'class5':5,'hiperekstensi':5,'hyperextension':5,
            '6':6,'class_6':6,'class6':6,'adduksi':6,'adduction':6,
        }
        for folder in sorted([d for d in data_dir.iterdir() if d.is_dir()]):
            label = label_map.get(folder.name.lower())
            if label is not None:
                for p in sorted(folder.glob('*.bmp')) + sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png')):
                    self.samples.append((str(p), label))
        if not self.samples:
            for p in sorted(data_dir.glob('*.bmp')):
                for k, v in label_map.items():
                    if k in p.name.lower():
                        self.samples.append((str(p), v)); break
        if not self.samples:
            raise ValueError(f"No images found in {data_dir}")
        print(f"  ViT Dataset: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label
    def get_labels(self): return [s[1] for s in self.samples]


# ── Custom ViT-Tiny (from scratch, for reference) ───────────────────────────

class PatchEmbedding(nn.Module):
    """Image → patch tokens."""
    def __init__(self, img_size=112, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)          # (B, E, H/P, W/P)
        B, E, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, N, E)


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""
    def __init__(self, embed_dim=192, n_heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTinyCustom(nn.Module):
    """
    ViT-Tiny from scratch (for educational comparison).
    img_size=112, patch_size=16 → 49 patches, embed_dim=192, depth=12, heads=3
    ⚠ NOT recommended for 863 samples without pretrained weights.
    """
    def __init__(self, img_size=112, patch_size=16, embed_dim=192,
                 depth=12, n_heads=3, num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.dropout(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)


def build_vit_pretrained(num_classes=NUM_CLASSES):
    """
    Load pretrained ViT-Tiny from timm library.
    Falls back to custom ViT-Tiny if timm not available.
    """
    try:
        import timm
        model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            num_classes=num_classes
        )
        print("  ✓ Loaded pretrained ViT-Tiny (timm)")
        return model, 'pretrained'
    except ImportError:
        print("  ⚠ timm not installed — using custom ViT-Tiny (not pretrained)")
        print("    Install timm: pip install timm")
        return ViTTinyCustom(), 'scratch'
    except Exception as e:
        print(f"  ⚠ timm error ({e}) — using custom ViT-Tiny")
        return ViTTinyCustom(), 'scratch'


def train_vit_fold(train_idx, val_idx, full_dataset,
                   epochs_warmup=5, epochs_finetune=30,
                   lr_head=1e-3, lr_full=2e-5, batch_size=16):
    """Two-phase ViT training."""
    class SubsetTransform(Dataset):
        def __init__(self, ds, idx, tfm):
            self.ds = ds; self.idx = idx; self.tfm = tfm
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            p, label = self.ds.samples[self.idx[i]]
            img = Image.open(p).convert('RGB')
            return self.tfm(img), label

    train_ds = SubsetTransform(full_dataset, train_idx, VIT_TRAIN_TRANSFORM)
    val_ds   = SubsetTransform(full_dataset, val_idx,   VIT_VAL_TRANSFORM)

    train_lbs = [full_dataset.samples[i][1] for i in train_idx]
    ctr = Counter(train_lbs); total = len(train_lbs)
    cw  = torch.FloatTensor([total/(NUM_CLASSES*ctr.get(i,1)) for i in range(NUM_CLASSES)]).to(DEVICE)

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    vl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model, mode = build_vit_pretrained()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Warm-up: head only (if pretrained)
    if mode == 'pretrained':
        for name, p in model.named_parameters():
            p.requires_grad = 'head' in name or 'classifier' in name
        head_params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.Adam(head_params, lr=lr_head)
        for _ in range(epochs_warmup):
            for imgs, lbs in tl:
                imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
                opt.zero_grad(); loss = criterion(model(imgs), lbs)
                loss.backward(); opt.step()

    # Full fine-tune
    for p in model.parameters(): p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=lr_full, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_finetune)
    best_acc, best_state = 0.0, None

    for epoch in range(epochs_finetune):
        model.train()
        for imgs, lbs in tl:
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(imgs), lbs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            preds_v, labels_v = [], []
            for imgs, lbs in vl:
                preds_v.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                labels_v.extend(lbs.numpy())
        acc = accuracy_score(labels_v, preds_v)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_v, labels_v = [], []
        for imgs, lbs in vl:
            preds_v.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            labels_v.extend(lbs.numpy())
    return np.array(preds_v), np.array(labels_v)


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: LIGHTWEIGHT SKELETON GCN
# ═══════════════════════════════════════════════════════════════════════════════

# MediaPipe 33-node skeleton adjacency
# Key connections for body pose
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,7),          # face-left
    (0,4),(4,5),(5,6),(6,8),          # face-right
    (9,10),                            # mouth
    (11,12),                           # shoulders
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),  # left arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),  # right arm
    (11,23),(12,24),(23,24),           # torso
    (23,25),(25,27),(27,29),(27,31),(29,31),  # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),  # right leg
]
NUM_NODES = 33
NODE_FEAT = 4   # x, y, z, visibility per keypoint


def build_adjacency(edges, n_nodes=NUM_NODES, self_loop=True):
    """Build normalized adjacency matrix A = D^{-1/2} A D^{-1/2}."""
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        np.fill_diagonal(A, 1.0)
    # Degree normalization
    D = np.diag(A.sum(axis=1) ** -0.5)
    A_norm = D @ A @ D
    return torch.FloatTensor(A_norm)


A_GLOBAL = build_adjacency(SKELETON_EDGES)


class GCNLayer(nn.Module):
    """Single GCN layer: H^{l+1} = σ(A H^l W)."""
    def __init__(self, in_feat, out_feat, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.linear = nn.Linear(in_feat, out_feat, bias=False)
        self.bn = nn.BatchNorm1d(out_feat)

    def forward(self, x):
        # x: (batch, nodes, in_feat)
        x = torch.bmm(self.adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        x = self.linear(x)
        B, N, F = x.shape
        x = self.bn(x.reshape(B*N, F)).reshape(B, N, F)
        return F.relu(x)


class SkeletonGCN(nn.Module):
    """
    2-layer GCN on 33-node MediaPipe skeleton.
    Input : (batch, 33, 4) — 33 nodes, 4 features each
    Output: (batch, num_classes)
    """
    def __init__(self, node_feat=NODE_FEAT, hidden=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.gcn1 = GCNLayer(node_feat, hidden,    A_GLOBAL)
        self.gcn2 = GCNLayer(hidden,    hidden*2,  A_GLOBAL)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden*2, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, nodes, feat)
        x = self.gcn1(x)
        x = self.gcn2(x)
        # Global mean pooling over nodes
        x = x.mean(dim=1)   # (batch, hidden*2)
        return self.classifier(x)


class SkeletonGraphDataset(Dataset):
    """
    Builds (33, 4) graph node features from MediaPipe keypoints.
    Falls back to random initialization if MediaPipe unavailable.
    """
    def __init__(self, data_dir, pose_model, augment=False):
        self.samples  = []
        self.augment  = augment
        data_dir = Path(data_dir)
        label_map = {
            '0':0,'class_0':0,'class0':0,'tidakada':0,'no_object':0,
            '1':1,'class_1':1,'class1':1,'unknown':1,
            '2':2,'class_2':2,'class2':2,'ekstensi':2,'extension':2,
            '3':3,'class_3':3,'class3':3,'fleksi':3,'flexion':3,
            '4':4,'class_4':4,'class4':4,'abduksi':4,'abduction':4,
            '5':5,'class_5':5,'class5':5,'hiperekstensi':5,'hyperextension':5,
            '6':6,'class_6':6,'class6':6,'adduksi':6,'adduction':6,
        }
        all_frames = []
        for folder in sorted([d for d in data_dir.iterdir() if d.is_dir()]):
            label = label_map.get(folder.name.lower())
            if label is not None:
                for p in sorted(list(folder.glob('*.bmp'))+list(folder.glob('*.jpg'))+list(folder.glob('*.png'))):
                    all_frames.append((p, label))
        if not all_frames:
            for p in sorted(data_dir.glob('*.bmp')):
                for k, v in label_map.items():
                    if k in p.name.lower():
                        all_frames.append((p, v)); break

        print(f"  Building skeleton graphs from {len(all_frames)} frames...")
        import cv2
        for i, (img_path, label) in enumerate(all_frames):
            if pose_model is not None:
                import mediapipe as mp
                img = cv2.imread(str(img_path))
                if img is None:
                    img = np.array(Image.open(str(img_path)).convert('RGB'))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                results = pose_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    kpts = []
                    for lm in results.pose_landmarks.landmark:
                        kpts.append([lm.x, lm.y, lm.z, lm.visibility])
                    graph = np.array(kpts, dtype=np.float32)  # (33, 4)
                else:
                    graph = np.zeros((NUM_NODES, NODE_FEAT), dtype=np.float32)
            else:
                # Fallback: simulate graph from pixel statistics
                img = np.array(Image.open(str(img_path)).convert('L'))
                h, w = img.shape
                # Sample 33 "pseudo-keypoints" from image grid
                xs = np.linspace(0, 1, 11)[1:-1]
                ys = np.linspace(0, 1, 5)[1:-1]
                pts = [(y, x) for y in ys for x in xs][:NUM_NODES]
                graph = np.array([
                    [pt[1], pt[0], float(img[int(pt[0]*h), int(pt[1]*w)])/255, 0.5]
                    for pt in pts
                ], dtype=np.float32)
                while len(graph) < NUM_NODES:
                    graph = np.vstack([graph, np.zeros((1, NODE_FEAT), dtype=np.float32)])

            self.samples.append((graph, label))
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(all_frames)}", end='\r')
        print(f"\n  ✓ Graph dataset ready ({len(self.samples)} graphs)")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        graph, label = self.samples[idx]
        t = torch.FloatTensor(graph)
        if self.augment:
            t = t + torch.randn_like(t) * 0.005
        return t, label
    def get_labels(self): return [s[1] for s in self.samples]


def train_gcn_fold(train_idx, val_idx, full_dataset, epochs=100, lr=1e-3):
    from torch.utils.data import Subset

    class AugSubset(Dataset):
        def __init__(self, ds, idx, aug):
            self.ds = ds; self.idx = idx; self.aug = aug
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            g, l = self.ds[self.idx[i]]
            if self.aug: g = g + torch.randn_like(g)*0.005
            return g, l

    train_ds = AugSubset(full_dataset, train_idx, True)
    val_ds   = AugSubset(full_dataset, val_idx,   False)

    train_lbs = [full_dataset.samples[i][1] for i in train_idx]
    ctr = Counter(train_lbs); total = len(train_lbs)
    cw  = torch.FloatTensor([total/(NUM_CLASSES*ctr.get(i,1)) for i in range(NUM_CLASSES)]).to(DEVICE)

    tl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    vl = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    model = SkeletonGCN().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss(weight=cw)
    best_acc, best_state, patience, no_imp = 0.0, None, 20, 0

    for epoch in range(epochs):
        model.train()
        for graphs, lbs in tl:
            graphs, lbs = graphs.to(DEVICE), lbs.to(DEVICE)
            opt.zero_grad(); loss = crit(model(graphs), lbs)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p, l = [], []
            for g, lb in vl:
                p.extend(model(g.to(DEVICE)).argmax(1).cpu().numpy())
                l.extend(lb.numpy())
        acc = accuracy_score(l, p)
        if acc > best_acc:
            best_acc = acc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp += 1
            if no_imp >= patience: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p, l = [], []
        for g, lb in vl:
            p.extend(model(g.to(DEVICE)).argmax(1).cpu().numpy())
            l.extend(lb.numpy())
    return np.array(p), np.array(l)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_transformer_gcn(data_dir, run_vit=True, run_gcn=True,
                        n_splits=5, output_dir='.'):
    print("\n" + "="*78)
    print("  VISION TRANSFORMER + GCN — S5-CV EVALUATION")
    print("="*78)

    csv_rows, all_results = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ── ViT ───────────────────────────────────────────────────────────────
    if run_vit:
        print("\n[ViT-Tiny]")
        ds = MotionFrameDataset(data_dir)
        all_labels = np.array(ds.get_labels())

        fold_accs, fold_f1s = [], []
        all_p, all_l = [], []
        t0 = time.time()

        for fold, (ti, vi) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            print(f"  Fold {fold+1}/{n_splits}  Train:{len(ti)} Val:{len(vi)}", end=' ')
            sys.stdout.flush()
            vp, vl = train_vit_fold(ti, vi, ds)
            acc = accuracy_score(vl, vp)*100
            f1  = f1_score(vl, vp, average='macro', zero_division=0)*100
            fold_accs.append(acc); fold_f1s.append(f1)
            all_p.extend(vp); all_l.extend(vl)
            print(f"→ Acc:{acc:.2f}% F1:{f1:.2f}%")

        elapsed = time.time()-t0
        mean_acc, std_acc, mean_f1 = np.mean(fold_accs), np.std(fold_accs), np.mean(fold_f1s)
        all_p, all_l = np.array(all_p), np.array(all_l)
        prec = precision_score(all_l, all_p, average='macro', zero_division=0)*100
        rec  = recall_score(all_l, all_p, average='macro', zero_division=0)*100
        cm   = confusion_matrix(all_l, all_p, labels=list(range(NUM_CLASSES)))

        print(f"\n  ViT-Tiny: Acc={mean_acc:.2f}%±{std_acc:.2f}% F1={mean_f1:.2f}% T={elapsed:.1f}s")
        print(f"  Per-Class:\n{classification_report(all_l, all_p, target_names=CLASS_NAMES, zero_division=0)}")
        print("  Confusion Matrix:")
        for i, row in enumerate(cm):
            r = row[i]/max(row.sum(),1)*100
            print(f"  {CLASS_NAMES[i]:10s}" + "".join(f"{v:6d}" for v in row) + f"  ← {r:.1f}%")

        all_results.append({'model':'ViT-Tiny','mean_acc':mean_acc,'std_acc':std_acc,
                            'mean_f1':mean_f1,'precision':prec,'recall':rec,'time_sec':elapsed})
        csv_rows.append({'Model':'ViT-Tiny(pretrained)','Accuracy_Mean(%)':f'{mean_acc:.4f}',
                         'Accuracy_Std(%)':f'{std_acc:.4f}','F1_Score(%)':f'{mean_f1:.4f}',
                         'Precision(%)':f'{prec:.4f}','Recall(%)':f'{rec:.4f}','Train_Time(s)':f'{elapsed:.1f}'})

    # ── GCN ───────────────────────────────────────────────────────────────
    if run_gcn:
        print("\n[Skeleton GCN]")
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                min_detection_confidence=0.3)
            print("  ✓ MediaPipe initialized for GCN")
        except ImportError:
            pose = None
            print("  ⚠ MediaPipe unavailable — using fallback pseudo-skeleton")

        ds_gcn = SkeletonGraphDataset(data_dir, pose)
        all_labels_gcn = np.array(ds_gcn.get_labels())

        fold_accs, fold_f1s = [], []
        all_p, all_l = [], []
        t0 = time.time()

        for fold, (ti, vi) in enumerate(skf.split(np.zeros(len(all_labels_gcn)), all_labels_gcn)):
            print(f"  Fold {fold+1}/{n_splits}  Train:{len(ti)} Val:{len(vi)}", end=' ')
            sys.stdout.flush()
            vp, vl = train_gcn_fold(ti, vi, ds_gcn)
            acc = accuracy_score(vl, vp)*100
            f1  = f1_score(vl, vp, average='macro', zero_division=0)*100
            fold_accs.append(acc); fold_f1s.append(f1)
            all_p.extend(vp); all_l.extend(vl)
            print(f"→ Acc:{acc:.2f}% F1:{f1:.2f}%")

        elapsed = time.time()-t0
        mean_acc, std_acc, mean_f1 = np.mean(fold_accs), np.std(fold_accs), np.mean(fold_f1s)
        all_p, all_l = np.array(all_p), np.array(all_l)
        prec = precision_score(all_l, all_p, average='macro', zero_division=0)*100
        rec  = recall_score(all_l, all_p, average='macro', zero_division=0)*100
        cm   = confusion_matrix(all_l, all_p, labels=list(range(NUM_CLASSES)))

        print(f"\n  Skeleton-GCN: Acc={mean_acc:.2f}%±{std_acc:.2f}% F1={mean_f1:.2f}% T={elapsed:.1f}s")
        print(f"  Per-Class:\n{classification_report(all_l, all_p, target_names=CLASS_NAMES, zero_division=0)}")

        all_results.append({'model':'Skeleton-GCN','mean_acc':mean_acc,'std_acc':std_acc,
                            'mean_f1':mean_f1,'precision':prec,'recall':rec,'time_sec':elapsed})
        csv_rows.append({'Model':'Skeleton-GCN','Accuracy_Mean(%)':f'{mean_acc:.4f}',
                         'Accuracy_Std(%)':f'{std_acc:.4f}','F1_Score(%)':f'{mean_f1:.4f}',
                         'Precision(%)':f'{prec:.4f}','Recall(%)':f'{rec:.4f}','Train_Time(s)':f'{elapsed:.1f}'})

    # Summary
    if all_results:
        print("\n" + "="*78)
        print("  SUMMARY — ViT + GCN")
        print("="*78)
        print(f"  {'Model':20s} {'Acc(%)':>10s} {'Std':>8s} {'F1(%)':>8s}")
        for r in all_results:
            star = ' ★' if r == max(all_results, key=lambda x: x['mean_acc']) else ''
            print(f"  {r['model']:20s} {r['mean_acc']:>10.2f} {r['std_acc']:>8.2f} {r['mean_f1']:>8.2f}{star}")

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'hasil_transformer_gcn_cv.csv')
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\n  ✓ Saved → {out_csv}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, default='.')
    parser.add_argument('--no_vit',     action='store_true')
    parser.add_argument('--no_gcn',     action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    run_transformer_gcn(
        data_dir=args.data_dir,
        run_vit=not args.no_vit,
        run_gcn=not args.no_gcn,
        output_dir=args.output_dir,
    )
