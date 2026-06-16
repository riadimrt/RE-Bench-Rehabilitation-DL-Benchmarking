"""
03_transformer_gcn_fixed.py
ViT-Tiny + Skeleton-GCN — Stratified 5-Fold CV
Compatible: MediaPipe >= 0.10.x  (Tasks API)

Usage:
  python 03_transformer_gcn_fixed.py --data_dir ./data --output_dir ./output
"""

import os, sys, time, argparse, warnings, urllib.request
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from PIL import Image

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE TASKS API  (v0.10.x)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

MODEL_ASSET_PATH = os.path.join(os.path.expanduser("~"), ".cache",
                                "mediapipe", "pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_full/float16/latest/"
             "pose_landmarker_full.task")
NUM_LANDMARKS = 33
FEAT_DIM      = NUM_LANDMARKS * 4   # x, y, z, visibility

CLASS_NAMES = {0:"Tidak Ada Objek",1:"Tdk Dikenal",2:"Ekstensi",
               3:"Fleksi",4:"Abduksi",5:"Hiperekstensi",6:"Adduksi"}
CLASS_ABBR  = ["NoObj","Unkn","Ext","Flex","Abd","HExt","Add"]


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def download_model_if_needed():
    if os.path.exists(MODEL_ASSET_PATH):
        return MODEL_ASSET_PATH
    os.makedirs(os.path.dirname(MODEL_ASSET_PATH), exist_ok=True)
    print(f"  [MediaPipe] Downloading pose model (~29 MB) → {MODEL_ASSET_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_ASSET_PATH,
            reporthook=lambda b,bs,t: print(
                f"\r  {min(100,int(b*bs*100/t))}%", end="", flush=True) if t>0 else None)
        print()
        return MODEL_ASSET_PATH
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        sys.exit(1)


def init_landmarker(model_path):
    """Create PoseLandmarker using Tasks API (MediaPipe >= 0.10.x)."""
    base_opts = mp_base.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def extract_keypoints(landmarker, img_path):
    """Extract 132-dim keypoint vector. Returns zeros on failure."""
    try:
        pil  = Image.open(img_path).convert("RGB")
        arr  = np.array(pil, dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = landmarker.detect(mp_img)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lmks = result.pose_landmarks[0]
            if len(lmks) == NUM_LANDMARKS:
                feat = []
                for lm in lmks:
                    feat.extend([lm.x, lm.y, lm.z, lm.visibility])
                return np.array(feat, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(FEAT_DIM, dtype=np.float32)


def extract_all_keypoints(paths):
    """Extract keypoints for all images. Returns (N, 132) array."""
    model_path = download_model_if_needed()
    print(f"  [MediaPipe] Extracting keypoints from {len(paths)} images...")
    landmarker = init_landmarker(model_path)
    X = np.zeros((len(paths), FEAT_DIM), dtype=np.float32)
    detected = 0
    for i, p in enumerate(paths):
        X[i] = extract_keypoints(landmarker, p)
        if X[i].sum() != 0:
            detected += 1
        if (i+1) % 100 == 0 or (i+1) == len(paths):
            print(f"\r  Progress: {i+1}/{len(paths)} | Detected: {detected}", end="", flush=True)
    landmarker.close()
    print(f"\n  Detection rate: {detected}/{len(paths)} = {detected/len(paths)*100:.1f}%")
    if detected < len(paths):
        print(f"  [WARN] {len(paths)-detected} images had no detection → zero vector used")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(data_dir):
    paths, labels = [], []
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    for cls_id in range(7):
        folder = Path(data_dir) / f"class_{cls_id}"
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in exts:
                paths.append(str(f))
                labels.append(cls_id)
    return paths, np.array(labels)


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths   = paths
        self.labels  = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_vit_transforms(train=True):
    import torchvision.transforms as T
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomAffine(degrees=0, translate=(0.1,0.1)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# VIT-TINY MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_vit(num_classes, device):
    try:
        import timm
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                   num_classes=num_classes)
        print("  ✓ Loaded pretrained ViT-Tiny (timm)")
        return model.to(device)
    except Exception as e:
        print(f"  [ERROR] ViT load failed: {e}")
        print("  pip install timm")
        return None


def compute_class_weights(y, num_classes, device):
    counts  = np.bincount(y, minlength=num_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_vit_fold(model, ds, tr_idx, val_idx, epochs, batch_size,
                   device, num_classes, patience=15):
    tr_ds = Subset(ds, tr_idx)
    val_ds = Subset(ds, val_idx)

    # Rebuild with train/val transforms
    paths  = ds.paths
    labels = ds.labels
    tr_set  = ImageDataset([paths[i] for i in tr_idx],
                            [labels[i] for i in tr_idx], get_vit_transforms(True))
    val_set = ImageDataset([paths[i] for i in val_idx],
                            [labels[i] for i in val_idx], get_vit_transforms(False))

    tr_loader  = DataLoader(tr_set,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    y_tr = labels[np.array(tr_idx)]
    class_w  = compute_class_weights(y_tr, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, patience_ctr = 0, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds_all.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                true_all.extend(yb.numpy())
        acc = accuracy_score(true_all, preds_all)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds_all.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            true_all.extend(yb.numpy())
    return np.array(preds_all), np.array(true_all)


# ─────────────────────────────────────────────────────────────────────────────
# SKELETON GCN
# ─────────────────────────────────────────────────────────────────────────────
# Human body skeleton adjacency (MediaPipe 33 landmarks)
BODY_EDGES = [
    (0,1),(1,2),(2,3),(3,7),          # head left
    (0,4),(4,5),(5,6),(6,8),          # head right
    (9,10),                            # mouth
    (11,12),                           # shoulders
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),  # left arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),  # right arm
    (11,23),(12,24),(23,24),           # torso
    (23,25),(25,27),(27,29),(27,31),(29,31),  # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),  # right leg
]

def build_adjacency(num_nodes=33, edges=BODY_EDGES):
    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Normalize (D^-1/2 A D^-1/2)
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-8)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))
        self.bn     = nn.BatchNorm1d(out_features)

    def forward(self, x, A):
        # x: (B, N, F)
        out = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)
        out = out @ self.weight + self.bias
        B, N, F = out.shape
        out = self.bn(out.reshape(B*N, F)).reshape(B, N, F)
        return torch.relu(out)


class SkeletonGCN(nn.Module):
    def __init__(self, in_features, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.gcn1    = GraphConvLayer(in_features, hidden)
        self.gcn2    = GraphConvLayer(hidden, hidden)
        self.gcn3    = GraphConvLayer(hidden, hidden // 2)
        self.dropout = nn.Dropout(dropout)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Sequential(
            nn.Linear(hidden // 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, A):
        # x: (B, N, F)
        x = self.dropout(self.gcn1(x, A))
        x = self.dropout(self.gcn2(x, A))
        x = self.gcn3(x, A)
        # Pool over nodes: (B, F, N) → (B, F)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)


def keypoints_to_graph(X_flat):
    """
    Convert (N, 132) flat keypoints to (N, 33, 4) graph node features.
    Each node = one landmark with (x, y, z, visibility).
    """
    N = X_flat.shape[0]
    return X_flat.reshape(N, NUM_LANDMARKS, 4)


def normalize_graph(X_tr, X_val):
    """Per-fold normalization without data leakage."""
    mn  = X_tr.min(axis=(0,1), keepdims=True)
    mx  = X_tr.max(axis=(0,1), keepdims=True)
    rng = np.where((mx - mn) < 1e-8, 1.0, mx - mn)
    return ((X_tr - mn) / rng).astype(np.float32), ((X_val - mn) / rng).astype(np.float32)


def train_gcn_fold(model, A, X_tr, y_tr, X_val, y_val,
                   epochs, batch_size, device, num_classes, patience=20):
    A = A.to(device)
    class_w  = compute_class_weights(y_tr, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.tensor(X_tr)   # (N, 33, 4)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_val).to(device)

    best_acc, best_state, patience_ctr = 0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start+batch_size]
            xb  = Xt[idx].to(device)
            yb  = yt[idx].to(device)
            optimizer.zero_grad()
            criterion(model(xb, A), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xv, A).argmax(1).cpu().numpy()
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xv, A).argmax(1).cpu().numpy()
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def print_fold_results(fold_metrics, model_name, elapsed):
    fm = np.array(fold_metrics)
    acc_m, acc_s = fm[:,0].mean(), fm[:,0].std()
    f1_m = fm[:,1].mean()
    prec_m = fm[:,2].mean()
    rec_m  = fm[:,3].mean()
    print(f"  {model_name}: Acc={acc_m*100:.2f}%±{acc_s*100:.2f}% "
          f"F1={f1_m*100:.2f}% T={elapsed:.1f}s")
    return {"acc":acc_m,"std":acc_s,"f1":f1_m,"prec":prec_m,"rec":rec_m,"time":elapsed}


def print_classification_details(all_true, all_preds, model_name):
    labels_arr = np.array(all_true)
    preds_arr  = np.array(all_preds)
    present    = sorted(np.unique(labels_arr).tolist())
    abbr       = [CLASS_ABBR[i] for i in present]

    print(f"  Per-Class ({model_name}):")
    print(classification_report(labels_arr, preds_arr,
          labels=present, target_names=abbr, zero_division=0, digits=2))

    cm = confusion_matrix(labels_arr, preds_arr, labels=present)
    print("  Confusion Matrix:")
    header = f"  {'':10s}" + "".join(f"{a:>7s}" for a in abbr)
    print(header)
    for i, row in enumerate(cm):
        rc = cm[i,i]/cm[i].sum() if cm[i].sum()>0 else 0
        print(f"  {abbr[i]:<10s}" + "".join(f"{v:7d}" for v in row)
              + f"  ← {rc*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_transformer_gcn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("  ⚠ GPU not detected. ViT training will be slow. Consider Google Colab.")

    print("=" * 78)
    print("  VISION TRANSFORMER + SKELETON GCN — S5-CV EVALUATION")
    print("=" * 78)

    paths, labels = load_dataset(args.data_dir)
    if len(paths) == 0:
        print(f"[ERROR] No images in {args.data_dir}")
        sys.exit(1)

    num_classes = len(np.unique(labels))
    os.makedirs(args.output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}

    # ── ViT-Tiny ──────────────────────────────────────────────────────────────
    if args.run_vit:
        print("\n[ViT-Tiny]")
        try:
            import timm
        except ImportError:
            print("  [ERROR] timm not installed. Run: pip install timm")
            args.run_vit = False

    if args.run_vit:
        full_ds    = ImageDataset(paths, labels)
        fold_metrics = []
        all_preds, all_true = [], []
        t0 = time.time()

        for fold, (tr_idx, val_idx) in enumerate(skf.split(paths, labels)):
            print(f"  Fold {fold+1}/5  Train:{len(tr_idx)} Val:{len(val_idx)}", end="  ")
            model = build_vit(num_classes, device)
            if model is None:
                break

            preds, true = train_vit_fold(model, full_ds, tr_idx, val_idx,
                                          args.epochs_vit, args.batch_size,
                                          device, num_classes)
            acc  = accuracy_score(true, preds)
            f1   = f1_score(true, preds, average="macro", zero_division=0)
            prec = precision_score(true, preds, average="macro", zero_division=0)
            rec  = recall_score(true, preds, average="macro", zero_division=0)
            fold_metrics.append((acc, f1, prec, rec))
            all_preds.extend(preds)
            all_true.extend(true)
            print(f"→ Acc:{acc*100:.2f}% F1:{f1*100:.2f}%")

        all_results["ViT-Tiny"] = print_fold_results(fold_metrics, "ViT-Tiny", time.time()-t0)
        print_classification_details(all_true, all_preds, "ViT-Tiny")

        csv_path = os.path.join(args.output_dir, "results_vit_tiny.csv")
        with open(csv_path, "w") as f:
            f.write("fold,acc,f1,prec,rec\n")
            for i, (a,f1v,p,r) in enumerate(fold_metrics):
                f.write(f"{i+1},{a:.4f},{f1v:.4f},{p:.4f},{r:.4f}\n")

    # ── Skeleton GCN ──────────────────────────────────────────────────────────
    if args.run_gcn:
        print("\n[Skeleton GCN]")
        if not MEDIAPIPE_OK:
            print("  [ERROR] mediapipe not installed. Run: pip install mediapipe")
        else:
            # Extract keypoints (reuse if already extracted)
            X_flat = extract_all_keypoints(paths)
            X_graph = keypoints_to_graph(X_flat)   # (N, 33, 4)
            A = build_adjacency()

            fold_metrics = []
            all_preds, all_true = [], []
            t0 = time.time()

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_flat, labels)):
                X_tr = X_graph[np.array(tr_idx)]
                X_val = X_graph[np.array(val_idx)]
                y_tr  = labels[np.array(tr_idx)]
                y_val = labels[np.array(val_idx)]

                X_tr_n, X_val_n = normalize_graph(X_tr, X_val)

                model = SkeletonGCN(in_features=4, num_classes=num_classes,
                                     hidden=args.gcn_hidden,
                                     dropout=args.gcn_dropout).to(device)

                preds = train_gcn_fold(model, A, X_tr_n, y_tr, X_val_n, y_val,
                                        args.epochs_gcn, args.batch_size,
                                        device, num_classes)

                acc  = accuracy_score(y_val, preds)
                f1   = f1_score(y_val, preds, average="macro", zero_division=0)
                prec = precision_score(y_val, preds, average="macro", zero_division=0)
                rec  = recall_score(y_val, preds, average="macro", zero_division=0)
                fold_metrics.append((acc, f1, prec, rec))
                all_preds.extend(preds)
                all_true.extend(y_val)
                print(f"  Fold {fold+1}/5  |  Train:{len(tr_idx)} Val:{len(val_idx)}"
                      f"  → Acc:{acc*100:.2f}% F1:{f1*100:.2f}%")

            all_results["Skeleton-GCN"] = print_fold_results(
                fold_metrics, "Skeleton-GCN", time.time()-t0)
            print_classification_details(all_true, all_preds, "Skeleton-GCN")

            csv_path = os.path.join(args.output_dir, "results_skeleton_gcn.csv")
            with open(csv_path, "w") as f:
                f.write("fold,acc,f1,prec,rec\n")
                for i, (a,f1v,p,r) in enumerate(fold_metrics):
                    f.write(f"{i+1},{a:.4f},{f1v:.4f},{p:.4f},{r:.4f}\n")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  FINAL COMPARISON — ViT + Skeleton-GCN (S5-CV)")
    print(f"{'='*78}")
    print(f"  {'Model':<20} {'Acc (%)':>10} {'Std':>8} {'F1 (%)':>8} "
          f"{'Prec':>8} {'Rec':>8} {'Time':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    best = max(all_results.values(), key=lambda x: x["acc"])["acc"] if all_results else 0
    for name, r in all_results.items():
        star = " ★" if r["acc"] == best else ""
        print(f"  {name:<20} {r['acc']*100:>10.2f} {r['std']*100:>8.2f} "
              f"{r['f1']*100:>8.2f} {r['prec']*100:>8.2f} "
              f"{r['rec']*100:>8.2f} {r['time']:>7.1f}s{star}")

    summary_path = os.path.join(args.output_dir, "summary_transformer_gcn.txt")
    with open(summary_path, "w") as f:
        f.write("Model,Acc,Std,F1,Prec,Rec,Time\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['acc']:.4f},{r['std']:.4f},{r['f1']:.4f},"
                    f"{r['prec']:.4f},{r['rec']:.4f},{r['time']:.1f}\n")
    print(f"\n  Summary saved → {summary_path}")
    print(f"{'='*78}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="ViT-Tiny + Skeleton-GCN S5-CV (MediaPipe >= 0.10.x)")
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--output_dir",  type=str, default="./output")
    p.add_argument("--no_vit",      action="store_true", help="Skip ViT-Tiny")
    p.add_argument("--no_gcn",      action="store_true", help="Skip Skeleton-GCN")
    p.add_argument("--epochs_vit",  type=int, default=30)
    p.add_argument("--epochs_gcn",  type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--gcn_hidden",  type=int, default=128)
    p.add_argument("--gcn_dropout", type=float, default=0.3)
    args = p.parse_args()
    args.run_vit = not args.no_vit
    args.run_gcn = not args.no_gcn
    return args


if __name__ == "__main__":
    args = parse_args()
    run_transformer_gcn(args)