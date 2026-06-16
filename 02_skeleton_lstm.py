"""
02_skeleton_lstm_fixed.py
MediaPipe Skeleton + LSTM/BiLSTM/Attn-LSTM — Stratified 5-Fold CV
Compatible: MediaPipe >= 0.10.x  (Tasks API)

Usage:
  python 02_skeleton_lstm_fixed.py --data_dir ./data --models lstm bilstm attn_lstm --seq_len 1 --output_dir ./output

Auto-downloads pose_landmarker_full.task model on first run.
"""

import os, sys, time, argparse, warnings, urllib.request
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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
    print("[WARN] mediapipe not installed. Run: pip install mediapipe")

MODEL_ASSET_PATH = os.path.join(os.path.expanduser("~"), ".cache",
                                "mediapipe", "pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_full/float16/latest/"
             "pose_landmarker_full.task")
NUM_LANDMARKS = 33
FEAT_DIM      = NUM_LANDMARKS * 4   # x, y, z, visibility


def download_model_if_needed():
    """Download pose_landmarker_full.task if not cached."""
    if os.path.exists(MODEL_ASSET_PATH):
        return MODEL_ASSET_PATH
    os.makedirs(os.path.dirname(MODEL_ASSET_PATH), exist_ok=True)
    print(f"  [MediaPipe] Downloading pose model (~29 MB) → {MODEL_ASSET_PATH}")
    print(f"  URL: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_ASSET_PATH,
            reporthook=lambda b,bs,t: print(
                f"\r  {min(100,int(b*bs*100/t))}%", end="", flush=True) if t>0 else None)
        print()
        return MODEL_ASSET_PATH
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print("  Manual download:")
        print(f"    curl -L '{MODEL_URL}' -o '{MODEL_ASSET_PATH}'")
        sys.exit(1)


def init_landmarker(model_path):
    """Create PoseLandmarker (Tasks API, IMAGE mode)."""
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
    """
    Extract 132-dim keypoint vector from image using Tasks API.
    Returns np.zeros(FEAT_DIM) on detection failure.
    """
    try:
        pil = Image.open(img_path).convert("RGB")
        arr = np.array(pil, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = landmarker.detect(mp_image)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lmks = result.pose_landmarks[0]   # first (only) person
            if len(lmks) == NUM_LANDMARKS:
                feat = []
                for lm in lmks:
                    feat.extend([lm.x, lm.y, lm.z, lm.visibility])
                return np.array(feat, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(FEAT_DIM, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Tidak Ada Objek",
    1: "Tdk Dikenal",
    2: "Ekstensi",
    3: "Fleksi",
    4: "Abduksi",
    5: "Hiperekstensi",
    6: "Adduksi",
}
CLASS_ABBR = ["NoObj","Unkn","Ext","Flex","Abd","HExt","Add"]


def load_dataset(data_dir):
    paths, labels = [], []
    data_dir = Path(data_dir)
    for cls_id in range(7):
        cls_folder = data_dir / f"class_{cls_id}"
        if not cls_folder.exists():
            continue
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        files = [f for f in cls_folder.iterdir() if f.suffix.lower() in exts]
        for f in files:
            paths.append(str(f))
            labels.append(cls_id)
    return paths, np.array(labels)


def print_distribution(labels, title="Class distribution"):
    print(f"  {title}:")
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for u, c in zip(unique, counts):
        name = CLASS_NAMES.get(int(u), f"Class {u}")
        bar = "█" * int(c / total * 30)
        print(f"    Class {u} ({name:<24}): {c:4d} ({c/total*100:5.1f}%) {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK MODELS
# ─────────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, seq_len, feat_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, F = x.shape
        x_bn = self.bn_input(x.reshape(B*S, F)).reshape(B, S, F)
        out, (h, _) = self.lstm(x_bn)
        feat = self.dropout(h[-1])
        return self.fc(feat)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, F = x.shape
        x_bn = self.bn_input(x.reshape(B*S, F)).reshape(B, S, F)
        out, (h, _) = self.lstm(x_bn)
        # concat forward + backward last hidden
        feat = self.dropout(torch.cat([h[-2], h[-1]], dim=1))
        return self.fc(feat)


class AttnLSTMClassifier(nn.Module):
    """LSTM with self-attention over time steps."""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, F = x.shape
        x_bn = self.bn_input(x.reshape(B*S, F)).reshape(B, S, F)
        out, _ = self.lstm(x_bn)            # (B, S, H)
        scores = torch.softmax(self.attn(out), dim=1)  # (B, S, 1)
        context = (scores * out).sum(dim=1)  # (B, H)
        feat = self.dropout(context)
        return self.fc(feat)


def build_model(model_name, input_dim, num_classes,
                hidden_dim=256, num_layers=2, dropout=0.3):
    name = model_name.lower()
    if name == "lstm":
        return LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif name == "bilstm":
        return BiLSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, dropout)
    elif name in ("attn_lstm", "attnlstm", "attention_lstm"):
        return AttnLSTMClassifier(input_dim, hidden_dim, num_layers, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION (per-fold, no leakage)
# ─────────────────────────────────────────────────────────────────────────────
def normalize(X_train, X_val):
    mn  = X_train.min(axis=0)
    mx  = X_train.max(axis=0)
    rng = np.where((mx - mn) < 1e-8, 1.0, mx - mn)
    X_train_n = (X_train - mn) / rng
    X_val_n   = (X_val   - mn) / rng
    return X_train_n.astype(np.float32), X_val_n.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(model, X_tr, y_tr, X_val, y_val,
                   epochs, batch_size, device, num_classes):
    model = model.to(device)
    class_w = compute_class_weights(y_tr, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.tensor(X_tr).unsqueeze(1)   # (N,1,F) — seq_len=1
    yt = torch.tensor(y_tr, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_acc, best_state, patience_ctr, patience = 0, None, 0, 20
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_val).unsqueeze(1).to(device)
            preds = model(Xv).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print("=" * 78)
    print("  MEDIAPIPE SKELETON + LSTM/BiLSTM/Attn-LSTM — S5-CV")
    print("=" * 78)
    print(f"  Data dir : {args.data_dir}")
    print(f"  Models   : {args.models}")
    print(f"  seq_len  : {args.seq_len}  (1 = per-frame, no temporal context)")
    print(f"  Epochs   : {args.epochs}")
    print("=" * 78)

    # ── 1. Load paths & labels ──
    paths, labels = load_dataset(args.data_dir)
    if len(paths) == 0:
        print(f"[ERROR] No images found in {args.data_dir}")
        print("  Expected structure: data/class_0/, data/class_1/, ..., data/class_6/")
        sys.exit(1)

    print(f"\n  [Dataset] {len(paths)} samples loaded")
    print_distribution(labels)
    num_classes = len(np.unique(labels))
    print(f"\n  [Info] Unique classes: {num_classes}")

    # ── 2. Download MediaPipe model ──
    model_path = download_model_if_needed()
    print(f"\n  [MediaPipe] Model: {model_path}")

    # ── 3. Extract keypoints for ALL images (one-time) ──
    print(f"\n  [MediaPipe] Extracting keypoints from {len(paths)} images...")
    print("  (This may take several minutes on CPU)")
    t0 = time.time()
    landmarker = init_landmarker(model_path)
    X = np.zeros((len(paths), FEAT_DIM), dtype=np.float32)
    detected, failed = 0, 0
    for i, p in enumerate(paths):
        X[i] = extract_keypoints(landmarker, p)
        if X[i].sum() != 0:
            detected += 1
        else:
            failed += 1
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"\r  Progress: {i+1}/{len(paths)} | "
                  f"Detected: {detected} | Failed: {failed}", end="", flush=True)
    landmarker.close()
    print(f"\n  Keypoint extraction: {time.time()-t0:.1f}s")
    print(f"  Detection rate: {detected}/{len(paths)} = {detected/len(paths)*100:.1f}%")
    if failed > 0:
        print(f"  [WARN] {failed} images had no detection → zero vector used")

    # ── 4. Per-model S5-CV ──
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for model_name in args.models:
        print(f"\n{'─'*78}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'─'*78}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        all_preds, all_true = [], []
        t_start = time.time()

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, labels)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = labels[tr_idx], labels[val_idx]

            # Normalize (per-fold, no leakage)
            X_tr_n, X_val_n = normalize(X_tr, X_val)

            # Build model
            model = build_model(model_name, FEAT_DIM, num_classes,
                                 hidden_dim=args.hidden_dim,
                                 num_layers=args.num_layers,
                                 dropout=args.dropout)

            # Train
            model = train_one_fold(model, X_tr_n, y_tr, X_val_n, y_val,
                                   args.epochs, args.batch_size, device, num_classes)

            # Evaluate
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val_n).unsqueeze(1).to(device)
                preds = model(Xv).argmax(dim=1).cpu().numpy()

            acc  = accuracy_score(y_val, preds)
            f1   = f1_score(y_val, preds, average="macro", zero_division=0)
            prec = precision_score(y_val, preds, average="macro", zero_division=0)
            rec  = recall_score(y_val, preds, average="macro", zero_division=0)
            fold_metrics.append((acc, f1, prec, rec))
            all_preds.extend(preds)
            all_true.extend(y_val)

            print(f"  Fold {fold+1}/5  |  Train: {len(tr_idx)} | Val: {len(val_idx)}"
                  f"  → Acc: {acc*100:.2f}%  F1: {f1*100:.2f}%")

        # Aggregate
        fm = np.array(fold_metrics)
        mean_acc  = fm[:,0].mean()
        std_acc   = fm[:,0].std()
        mean_f1   = fm[:,1].mean()
        mean_prec = fm[:,2].mean()
        mean_rec  = fm[:,3].mean()
        elapsed   = time.time() - t_start

        all_results[model_name] = {
            "acc": mean_acc, "std": std_acc, "f1": mean_f1,
            "prec": mean_prec, "rec": mean_rec, "time": elapsed,
        }

        print(f"  {'─'*60}")
        print(f"  {model_name.upper()} RESULTS (S5-CV)")
        print(f"  Accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  F1 Macro : {mean_f1*100:.2f}%")
        print(f"  Precision: {mean_prec*100:.2f}%  Recall: {mean_rec*100:.2f}%")
        print(f"  Time     : {elapsed:.1f}s")
        print(f"  {'─'*60}")

        # Per-class report
        from sklearn.metrics import classification_report
        labels_arr = np.array(all_true)
        preds_arr  = np.array(all_preds)
        present = sorted(np.unique(labels_arr).tolist())
        abbr    = [CLASS_ABBR[i] for i in present]
        print("  Per-Class Report:")
        print(classification_report(labels_arr, preds_arr,
              labels=present, target_names=abbr, zero_division=0,
              digits=2))

        # Confusion matrix
        cm = confusion_matrix(labels_arr, preds_arr, labels=present)
        print("  Confusion Matrix:")
        header = f"{'':24s}" + "".join(f"{a:>8s}" for a in abbr)
        print(f"  {header}")
        for i, row in enumerate(cm):
            recall_i = cm[i,i]/cm[i].sum() if cm[i].sum()>0 else 0
            row_str = "".join(f"{v:8d}" for v in row)
            print(f"  {abbr[i]:<24s}{row_str}  ← Recall={recall_i*100:.1f}%")

        # Save CSV
        csv_path = os.path.join(args.output_dir, f"results_skeleton_{model_name}.csv")
        with open(csv_path, "w") as f:
            f.write("fold,acc,f1,prec,rec\n")
            for i, (a,f1v,p,r) in enumerate(fold_metrics):
                f.write(f"{i+1},{a:.4f},{f1v:.4f},{p:.4f},{r:.4f}\n")
        print(f"  Results saved → {csv_path}")

    # ── 5. Final comparison ──
    print(f"\n{'='*78}")
    print("  FINAL COMPARISON — SKELETON LSTM/BiLSTM/Attn-LSTM (S5-CV)")
    print(f"{'='*78}")
    header = f"  {'Model':<20} {'Acc (%)':>10} {'Std':>8} {'F1 (%)':>8} {'Prec':>8} {'Rec':>8} {'Time':>8}"
    print(header)
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    best_acc = max(v["acc"] for v in all_results.values())
    for name, r in all_results.items():
        star = " ★" if r["acc"] == best_acc else ""
        print(f"  {name:<20} {r['acc']*100:>10.2f} {r['std']*100:>8.2f} "
              f"{r['f1']*100:>8.2f} {r['prec']*100:>8.2f} "
              f"{r['rec']*100:>8.2f} {r['time']:>7.1f}s{star}")
    print(f"{'='*78}")

    # Save final summary
    summary_path = os.path.join(args.output_dir, "summary_skeleton_lstm.txt")
    with open(summary_path, "w") as f:
        f.write("Model,Acc,Std,F1,Prec,Rec,Time\n")
        for name, r in all_results.items():
            f.write(f"{name},{r['acc']:.4f},{r['std']:.4f},{r['f1']:.4f},"
                    f"{r['prec']:.4f},{r['rec']:.4f},{r['time']:.1f}\n")
    print(f"\n  Summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="MediaPipe Skeleton + LSTM S5-CV (MediaPipe >= 0.10.x)")
    p.add_argument("--data_dir",   type=str, default="./data",
                   help="Root folder with class_0 .. class_6 subfolders")
    p.add_argument("--models",     nargs="+",
                   default=["lstm", "bilstm", "attn_lstm"],
                   choices=["lstm", "bilstm", "attn_lstm"],
                   help="Models to evaluate")
    p.add_argument("--seq_len",    type=int, default=1,
                   help="Sequence length (1 = per-frame static mode)")
    p.add_argument("--epochs",     type=int, default=80,
                   help="Max training epochs per fold (early stopping applies)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--output_dir", type=str, default="./output")
    return p.parse_args()


if __name__ == "__main__":
    if not MEDIAPIPE_OK:
        print("[ERROR] mediapipe not installed.")
        print("  pip install mediapipe")
        sys.exit(1)
    args = parse_args()
    run_evaluation(args)
