"""
=============================================================================
  KONFIGURASI GLOBAL — Modern Deep Learning untuk Human Body Motion
  Comparative Study: Classical ML vs. Modern DL
=============================================================================
  Dataset   : 863 BMP frames, 7 kelas gerakan rehabilitasi
  Classical  : RF+ChainCode = 88.07% CV, 70.65% Video (baseline)
  Modern     : CNN / CNN+BiLSTM / Skeleton-Transformer-GCN
=============================================================================
"""

import os

# ── PATH DATASET ──────────────────────────────────────────────────────────
# Sesuaikan dengan struktur folder dataset Anda:
# dataset/
#   class_0_NoObject/     ← 55 frame
#   class_1_Unknown/      ← 32 frame
#   class_2_Extension/    ← 374 frame
#   class_3_Flexion/      ← 138 frame
#   class_4_Abduction/    ← 158 frame
#   class_5_Hyperext/     ← 62 frame
#   class_6_Adduction/    ← 44 frame

DATASET_ROOT   = "./dataset"          # ← ganti sesuai lokasi Anda
VIDEO_PATH     = "./videos"           # ← folder video .wmv
OUTPUT_DIR     = "./results_modern"   # output CSV, model, plots

# ── KELAS ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "No Object",      # 0
    "Unknown",        # 1
    "Extension",      # 2
    "Flexion",        # 3
    "Abduction",      # 4
    "Hyperextension", # 5
    "Adduction",      # 6
]
NUM_CLASSES = 7

# ── TRAINING ──────────────────────────────────────────────────────────────
SEED          = 42
N_FOLDS       = 5        # Stratified K-Fold (sama dg klasik)
BATCH_SIZE    = 16       # kecil karena dataset kecil
EPOCHS_CNN    = 50
EPOCHS_LSTM   = 40
EPOCHS_GCN    = 60
LR            = 1e-4
WEIGHT_DECAY  = 1e-4

# ── IMAGE ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224      # MobileNetV2 input
SEQUENCE_LEN  = 8        # frame per sequence untuk LSTM

# ── SKELETON (MediaPipe Pose) ─────────────────────────────────────────────
N_LANDMARKS   = 33       # MediaPipe Pose keypoints
LANDMARK_DIM  = 3        # x, y, visibility
SKELETON_FEAT = N_LANDMARKS * LANDMARK_DIM   # 99 features

# ── GCN adjacency (MediaPipe Pose body connections) ───────────────────────
# Ref: https://google.github.io/mediapipe/solutions/pose.html
SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(3,7),       # face
    (0,4),(4,5),(5,6),(6,8),
    (9,10),                         # mouth
    (11,12),                        # shoulders
    (11,13),(13,15),               # left arm
    (12,14),(14,16),               # right arm
    (11,23),(12,24),(23,24),       # torso
    (23,25),(25,27),(27,29),(29,31),(27,31),  # left leg
    (24,26),(26,28),(28,30),(30,32),(28,32),  # right leg
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[Config] Dataset: {DATASET_ROOT}")
print(f"[Config] Classes: {NUM_CLASSES}, Seed: {SEED}, Folds: {N_FOLDS}")
