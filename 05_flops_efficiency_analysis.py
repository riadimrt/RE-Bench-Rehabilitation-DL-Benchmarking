"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODULE 5: ENERGY-AWARE EFFICIENCY ANALYSIS                                  ║
║  FLOPs, Parameters, Model Size, Inference Latency                           ║
║  Inspired by: Energy-Aware GPU Skinning (Shalaby et al.)                    ║
║                                                                              ║
║  Tujuan:                                                                     ║
║  Menganalisis efisiensi komputasi semua model dari perspektif                ║
║  energy-aware computing — mengadaptasi prinsip dari GPU Skinning            ║
║  ke domain AI inference untuk rehabilitasi klinis                           ║
║                                                                              ║
║  Output:                                                                     ║
║    - efficiency_analysis.csv                                                 ║
║    - inference_latency.csv                                                   ║
║    - energy_aware_report.txt                                                 ║
║    - efficiency_chart.png                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    pip install thop
    python 05_flops_efficiency_analysis.py --output_dir ./output
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import models

# ─────────────────────────────────────────────────────────────────────────────
# HASIL S5-CV YANG SUDAH ADA (DATA VALID DARI EKSPERIMEN)
# ─────────────────────────────────────────────────────────────────────────────
S5CV_RESULTS = {
    'MobileNetV2': {
        'accuracy': 97.22, 'std': 1.12, 'f1': 96.00,
        'precision': 97.74, 'recall': 95.12,
        's5cv_time_s': 2187.7, 'paradigm': 'CNN',
        'gpu_required': True, 'cpu_native': False,
        'relative_speed': 1.0,
    },
    'EfficientNet-B0': {
        'accuracy': 96.87, 'std': 1.49, 'f1': 95.54,
        'precision': 96.99, 'recall': 94.99,
        's5cv_time_s': 3057.7, 'paradigm': 'CNN',
        'gpu_required': True, 'cpu_native': False,
        'relative_speed': 2187.7/3057.7,
    },
    'ViT-Tiny': {
        'accuracy': 97.22, 'std': 1.18, 'f1': 96.73,
        'precision': 97.43, 'recall': 96.44,
        's5cv_time_s': 6335.7, 'paradigm': 'Transformer',
        'gpu_required': True, 'cpu_native': False,
        'relative_speed': 2187.7/6335.7,
    },
    'BiLSTM': {
        'accuracy': 95.13, 'std': 2.10, 'f1': 93.15,
        'precision': 94.45, 'recall': 93.46,
        's5cv_time_s': 62.9, 'paradigm': 'Sequential',
        'gpu_required': False, 'cpu_native': True,
        'relative_speed': 2187.7/62.9,
    },
    'Attn-LSTM': {
        'accuracy': 95.13, 'std': 1.67, 'f1': 92.87,
        'precision': 93.57, 'recall': 93.33,
        's5cv_time_s': 33.8, 'paradigm': 'Sequential',
        'gpu_required': False, 'cpu_native': True,
        'relative_speed': 2187.7/33.8,
    },
    'LSTM': {
        'accuracy': 94.90, 'std': 1.57, 'f1': 92.05,
        'precision': 93.64, 'recall': 92.23,
        's5cv_time_s': 29.6, 'paradigm': 'Sequential',
        'gpu_required': False, 'cpu_native': True,
        'relative_speed': 2187.7/29.6,
    },
    'Skeleton-GCN': {
        'accuracy': 94.78, 'std': 1.80, 'f1': 92.71,
        'precision': 94.56, 'recall': 92.46,
        's5cv_time_s': 90.7, 'paradigm': 'Graph',
        'gpu_required': False, 'cpu_native': True,
        'relative_speed': 2187.7/90.7,
    },
}

NUM_CLASSES = 7
IMG_INPUT   = torch.randn(1, 3, 224, 224)
SEQ_INPUT   = torch.randn(1, 1, 132)   # skeleton 132-dim


# ─────────────────────────────────────────────────────────────────────────────
# BANGUN MODEL (TANPA TRAINING)
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenetv2(num_classes=NUM_CLASSES):
    model = models.mobilenet_v2(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
    return model, IMG_INPUT

def build_efficientnet(num_classes=NUM_CLASSES):
    model = models.efficientnet_b0(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
    return model, IMG_INPUT

def build_resnet18(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
    return model, IMG_INPUT

def build_vit_tiny(num_classes=NUM_CLASSES):
    try:
        import timm
        model = timm.create_model('vit_tiny_patch16_224',
                                   pretrained=False, num_classes=num_classes)
        return model, IMG_INPUT
    except ImportError:
        print("  [WARN] timm not installed → ViT-Tiny skipped")
        return None, None

def build_convnext_tiny(num_classes=NUM_CLASSES):
    """ConvNeXt-Tiny — CNN terbaru 2022, NOVELTY vs Shalaby"""
    try:
        import timm
        model = timm.create_model('convnext_tiny',
                                   pretrained=False, num_classes=num_classes)
        return model, IMG_INPUT
    except ImportError:
        print("  [WARN] timm not installed → ConvNeXt-Tiny skipped")
        return None, None

def build_efficientnetv2(num_classes=NUM_CLASSES):
    """EfficientNetV2-S — lebih baru dari EfficientNet-B0"""
    try:
        import timm
        model = timm.create_model('efficientnetv2_s',
                                   pretrained=False, num_classes=num_classes)
        return model, IMG_INPUT
    except ImportError:
        print("  [WARN] timm not installed → EfficientNetV2-S skipped")
        return None, None

# Sequential models (LSTM-based)
class LSTMModel(nn.Module):
    def __init__(self, input_dim=132, hidden=256, layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, dropout=dropout if layers>1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        B, S, F = x.shape
        x = self.bn(x.reshape(B*S, F)).reshape(B, S, F)
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(h[-1]))

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=132, hidden=256, layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers>1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        B, S, F = x.shape
        x = self.bn(x.reshape(B*S, F)).reshape(B, S, F)
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))

class AttnLSTMModel(nn.Module):
    def __init__(self, input_dim=132, hidden=256, layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers>1 else 0)
        self.attn = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        B, S, F = x.shape
        x = self.bn(x.reshape(B*S, F)).reshape(B, S, F)
        out, _ = self.lstm(x)
        scores = torch.softmax(self.attn(out), dim=1)
        return self.fc(self.drop((scores * out).sum(dim=1)))

# GCN
class SkeletonGCN(nn.Module):
    def __init__(self, in_features=4, hidden=128,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.gc1 = nn.Linear(in_features, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.gc3 = nn.Linear(hidden, hidden//2)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden//2, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        # x: (B, 33, 4)
        x = torch.relu(self.gc1(x))
        x = self.drop(x)
        x = torch.relu(self.gc2(x))
        x = self.drop(x)
        x = torch.relu(self.gc3(x))
        x = self.pool(x.transpose(1,2)).squeeze(-1)
        return self.fc(x)

GCN_INPUT = torch.randn(1, 33, 4)

MODEL_BUILDERS = {
    'MobileNetV2':    build_mobilenetv2,
    'EfficientNet-B0': build_efficientnet,
    'ResNet18':       build_resnet18,
    'ViT-Tiny':       build_vit_tiny,
    'ConvNeXt-Tiny':  build_convnext_tiny,
    'EfficientNetV2': build_efficientnetv2,
}

SEQUENTIAL_MODELS = {
    'LSTM':         (LSTMModel(), SEQ_INPUT),
    'BiLSTM':       (BiLSTMModel(), SEQ_INPUT),
    'Attn-LSTM':    (AttnLSTMModel(), SEQ_INPUT),
    'Skeleton-GCN': (SkeletonGCN(), GCN_INPUT),
}


# ─────────────────────────────────────────────────────────────────────────────
# HITUNG METRICS EFISIENSI
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    """Hitung jumlah parameter (total dan trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    """Hitung ukuran model dalam MB."""
    total_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size()
                       for b in model.buffers())
    return total_bytes / (1024 ** 2)

def compute_flops(model, input_tensor):
    """Hitung FLOPs menggunakan thop library."""
    try:
        from thop import profile, clever_format
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flops, params = profile(model, inputs=(input_tensor,),
                                    verbose=False)
        flops_g = flops / 1e9
        return flops_g, params
    except ImportError:
        print("  [INFO] thop not found → using parameter-based estimate")
        print("         Install: pip install thop")
        # Estimasi kasar berdasarkan jumlah parameter
        total_p, _ = count_params(model)
        flops_est = total_p * 2 / 1e9  # rule of thumb: ~2 FLOPs per param
        return flops_est, total_p
    except Exception as e:
        total_p, _ = count_params(model)
        flops_est = total_p * 2 / 1e9
        return flops_est, total_p

def measure_inference_latency(model, input_tensor, n_runs=50):
    """
    Ukur latensi inferensi CPU (single sample).
    Ini BERBEDA dari S5-CV time — ini adalah per-sample latency.
    """
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps_cpu': 1000 / np.mean(times),
    }

def compute_energy_efficiency_score(accuracy, flops_g, model_size_mb,
                                     latency_ms):
    """
    Hitung Energy Efficiency Score (EES) — metrik gabungan.
    Terinspirasi dari energy-aware computing (Shalaby Art.3).

    EES = Accuracy / (FLOPs × Latency × ModelSize)^(1/3)
    Nilai lebih tinggi = lebih efisien secara energi
    """
    try:
        denom = (flops_g * latency_ms * model_size_mb) ** (1/3)
        if denom == 0:
            return 0
        return (accuracy / 100) / denom * 100
    except:
        return 0

def compute_clinical_deployment_score(accuracy, latency_ms,
                                       cpu_native, model_size_mb):
    """
    Hitung Clinical Deployment Score (CDS).
    Menggabungkan akurasi + kecepatan + portabilitas.
    """
    acc_norm  = accuracy / 100
    speed_norm = min(1.0, 100 / max(latency_ms, 1))
    port_bonus = 0.2 if cpu_native else 0.0
    size_norm  = min(1.0, 50 / max(model_size_mb, 1))
    cds = (0.5 * acc_norm +
           0.25 * speed_norm +
           0.15 * size_norm +
           0.10 * (acc_norm + port_bonus))
    return min(1.0, cds) * 100


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_efficiency_analysis(output_dir='./output', n_runs=50):
    print("\n" + "="*78)
    print("  ENERGY-AWARE EFFICIENCY ANALYSIS")
    print("  Inspired by: Energy-Aware GPU Skinning (Shalaby et al.)")
    print("="*78)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    # ── CNN + Transformer Models ──────────────────────────────────────────
    print("\n  [PHASE 1] CNN & Transformer Models")
    print("  " + "─"*60)

    for name, builder in MODEL_BUILDERS.items():
        print(f"\n  → Analyzing: {name}")
        result = builder()
        if result[0] is None:
            print(f"    ⚠ Skipped (model not available)")
            continue
        model, inp = result

        model.eval()

        # Metrics
        total_p, train_p = count_params(model)
        size_mb = get_model_size_mb(model)
        flops_g, _ = compute_flops(model, inp)
        latency = measure_inference_latency(model, inp, n_runs)

        # S5-CV data (jika ada)
        s5cv = S5CV_RESULTS.get(name, {})
        accuracy  = s5cv.get('accuracy', 0)
        s5cv_time = s5cv.get('s5cv_time_s', 0)
        paradigm  = s5cv.get('paradigm', 'CNN')
        cpu_nat   = s5cv.get('cpu_native', False)

        ees = compute_energy_efficiency_score(
            accuracy, flops_g, size_mb, latency['mean_ms'])
        cds = compute_clinical_deployment_score(
            accuracy, latency['mean_ms'], cpu_nat, size_mb)

        print(f"    Params:   {total_p/1e6:.2f}M")
        print(f"    Size:     {size_mb:.2f} MB")
        print(f"    FLOPs:    {flops_g:.3f} GFLOPs")
        print(f"    Latency:  {latency['mean_ms']:.2f} ± "
              f"{latency['std_ms']:.2f} ms")
        print(f"    FPS(CPU): {latency['fps_cpu']:.1f}")
        if accuracy > 0:
            print(f"    Accuracy: {accuracy:.2f}% (from S5-CV)")
            print(f"    EES:      {ees:.3f}")
            print(f"    CDS:      {cds:.2f}")

        results.append({
            'Model': name,
            'Paradigm': paradigm,
            'Params_M': round(total_p/1e6, 2),
            'Params_Trainable_M': round(train_p/1e6, 2),
            'Model_Size_MB': round(size_mb, 2),
            'FLOPs_G': round(flops_g, 3),
            'Latency_Mean_ms': round(latency['mean_ms'], 2),
            'Latency_Std_ms': round(latency['std_ms'], 2),
            'FPS_CPU': round(latency['fps_cpu'], 1),
            'S5CV_Time_s': s5cv_time,
            'Accuracy_%': accuracy,
            'Std_%': s5cv.get('std', 0),
            'F1_%': s5cv.get('f1', 0),
            'CPU_Native': cpu_nat,
            'GPU_Required': s5cv.get('gpu_required', True),
            'Relative_Speed': round(s5cv.get('relative_speed', 0), 1),
            'EES': round(ees, 3),
            'CDS': round(cds, 2),
        })

    # ── Sequential + GCN Models ───────────────────────────────────────────
    print("\n  [PHASE 2] Sequential & Graph Models")
    print("  " + "─"*60)

    for name, (model, inp) in SEQUENTIAL_MODELS.items():
        print(f"\n  → Analyzing: {name}")
        model.eval()

        total_p, train_p = count_params(model)
        size_mb = get_model_size_mb(model)
        flops_g, _ = compute_flops(model, inp)
        latency = measure_inference_latency(model, inp, n_runs)

        s5cv = S5CV_RESULTS.get(name, {})
        accuracy  = s5cv.get('accuracy', 0)
        s5cv_time = s5cv.get('s5cv_time_s', 0)
        paradigm  = s5cv.get('paradigm', 'Sequential')
        cpu_nat   = s5cv.get('cpu_native', True)

        ees = compute_energy_efficiency_score(
            accuracy, max(flops_g, 0.001), size_mb, latency['mean_ms'])
        cds = compute_clinical_deployment_score(
            accuracy, latency['mean_ms'], cpu_nat, size_mb)

        print(f"    Params:   {total_p/1e6:.3f}M")
        print(f"    Size:     {size_mb:.2f} MB")
        print(f"    FLOPs:    {flops_g:.4f} GFLOPs")
        print(f"    Latency:  {latency['mean_ms']:.2f} ± "
              f"{latency['std_ms']:.2f} ms")
        print(f"    FPS(CPU): {latency['fps_cpu']:.1f}")
        if accuracy > 0:
            print(f"    Accuracy: {accuracy:.2f}% (from S5-CV)")
            print(f"    EES:      {ees:.3f}")
            print(f"    CDS:      {cds:.2f}")

        results.append({
            'Model': name,
            'Paradigm': paradigm,
            'Params_M': round(total_p/1e6, 3),
            'Params_Trainable_M': round(train_p/1e6, 3),
            'Model_Size_MB': round(size_mb, 2),
            'FLOPs_G': round(flops_g, 4),
            'Latency_Mean_ms': round(latency['mean_ms'], 2),
            'Latency_Std_ms': round(latency['std_ms'], 2),
            'FPS_CPU': round(latency['fps_cpu'], 1),
            'S5CV_Time_s': s5cv_time,
            'Accuracy_%': accuracy,
            'Std_%': s5cv.get('std', 0),
            'F1_%': s5cv.get('f1', 0),
            'CPU_Native': cpu_nat,
            'GPU_Required': s5cv.get('gpu_required', False),
            'Relative_Speed': round(s5cv.get('relative_speed', 0), 1),
            'EES': round(ees, 3),
            'CDS': round(cds, 2),
        })

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'efficiency_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Efficiency data saved → {csv_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLES
# ─────────────────────────────────────────────────────────────────────────────

def print_efficiency_tables(df, output_dir):
    print("\n" + "="*90)
    print("  TABLE: ENERGY-AWARE EFFICIENCY ANALYSIS — ALL MODELS")
    print("="*90)

    # Sort by accuracy desc
    df_sorted = df.sort_values('Accuracy_%', ascending=False)

    print(f"\n  {'Model':<18} {'Para':<8} {'Size':>7} {'FLOPs':>8} "
          f"{'Lat(ms)':>9} {'FPS':>7} {'Acc%':>7} {'EES':>7} {'CDS':>7} {'CPU?':>6}")
    print(f"  {'─'*18} {'─'*8} {'─'*7} {'─'*8} {'─'*9} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*6}")

    for _, row in df_sorted.iterrows():
        cpu_str = '✅' if row['CPU_Native'] else '❌GPU'
        acc_str = f"{row['Accuracy_%']:.2f}" if row['Accuracy_%'] > 0 else 'N/A'
        print(f"  {row['Model']:<18} {row['Params_M']:<8} "
              f"{row['Model_Size_MB']:>6.1f}MB "
              f"{row['FLOPs_G']:>7.3f}G "
              f"{row['Latency_Mean_ms']:>8.1f}ms "
              f"{row['FPS_CPU']:>6.1f} "
              f"{acc_str:>7} "
              f"{row['EES']:>7.3f} "
              f"{row['CDS']:>7.2f} "
              f"{cpu_str:>6}")

    # ── Energy Efficiency Ranking ─────────────────────────────────────────
    print("\n" + "="*90)
    print("  ENERGY EFFICIENCY RANKING (terinspirasi GPU Skinning principles)")
    print("="*90)
    print(f"\n  Rank  {'Model':<18} {'EES':>8} {'Class':>12} {'Clinical Insight'}")
    print(f"  {'─'*5} {'─'*18} {'─'*8} {'─'*12} {'─'*30}")

    df_ees = df[df['Accuracy_%'] > 0].sort_values('EES', ascending=False)
    for rank, (_, row) in enumerate(df_ees.iterrows(), 1):
        hw  = 'CPU-Native' if row['CPU_Native'] else 'GPU-Required'
        insight = _get_clinical_insight(row['Model'])
        star = ' ★' if rank == 1 else ''
        print(f"  {rank:<5} {row['Model']:<18} {row['EES']:>8.3f} "
              f"{hw:>12} {insight}{star}")

    # ── Computational Bifurcation ─────────────────────────────────────────
    print("\n" + "="*90)
    print("  COMPUTATIONAL BIFURCATION ANALYSIS")
    print("  (Inspired by CPU-GPU workload separation — Shalaby GPU Skinning)")
    print("="*90)

    gpu_class = df[df['GPU_Required'] == True]
    cpu_class = df[df['CPU_Native'] == True]

    print(f"\n  GPU-CLASS ARCHITECTURES ({len(gpu_class)} models):")
    print(f"  → Require dedicated GPU for real-time clinical deployment")
    for _, row in gpu_class.iterrows():
        print(f"     {row['Model']:<18} FLOPs:{row['FLOPs_G']:.3f}G  "
              f"Size:{row['Model_Size_MB']:.1f}MB  "
              f"Latency:{row['Latency_Mean_ms']:.1f}ms")

    print(f"\n  CPU-NATIVE ARCHITECTURES ({len(cpu_class)} models):")
    print(f"  → Fully operational without GPU acceleration")
    for _, row in cpu_class.iterrows():
        speed = S5CV_RESULTS.get(row['Model'], {}).get('relative_speed', 0)
        print(f"     {row['Model']:<18} FLOPs:{row['FLOPs_G']:.4f}G  "
              f"Size:{row['Model_Size_MB']:.1f}MB  "
              f"Latency:{row['Latency_Mean_ms']:.1f}ms  "
              f"({speed:.1f}× faster than MobileNetV2)")

    # ── Save report ───────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, 'energy_aware_report.txt')
    with open(report_path, 'w') as f:
        f.write("Model,Paradigm,Params_M,Size_MB,FLOPs_G,")
        f.write("Latency_ms,FPS_CPU,Accuracy_%,EES,CDS,CPU_Native\n")
        for _, row in df.iterrows():
            f.write(f"{row['Model']},{row['Paradigm']},{row['Params_M']},"
                    f"{row['Model_Size_MB']},{row['FLOPs_G']},"
                    f"{row['Latency_Mean_ms']},{row['FPS_CPU']},"
                    f"{row['Accuracy_%']},{row['EES']},{row['CDS']},"
                    f"{row['CPU_Native']}\n")
    print(f"\n  ✓ Report saved → {report_path}")


def _get_clinical_insight(model_name):
    insights = {
        'MobileNetV2':    'Best accuracy, GPU needed',
        'EfficientNet-B0':'Good accuracy, moderate cost',
        'ResNet18':       'Baseline CNN, GPU needed',
        'ViT-Tiny':       'Best ambiguous pose recall',
        'ConvNeXt-Tiny':  'Latest CNN architecture (2022)',
        'EfficientNetV2': 'Improved EfficientNet (2021)',
        'Attn-LSTM':      'Best CPU efficiency overall',
        'BiLSTM':         'Bidirectional temporal context',
        'LSTM':           'Lightest model, ARM-compatible',
        'Skeleton-GCN':   'Anatomically interpretable',
    }
    return insights.get(model_name, '—')


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def make_efficiency_charts(df, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        df_valid = df[df['Accuracy_%'] > 0].copy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            'Energy-Aware Efficiency Analysis\n'
            'Inspired by: Energy-Aware GPU Skinning Principles (Shalaby et al.)',
            fontsize=13, fontweight='bold')

        colors = {
            'CNN': '#2563EB', 'Transformer': '#9333EA',
            'Sequential': '#16A34A', 'Graph': '#DC2626'
        }
        bar_colors = [colors.get(p, '#6B7280')
                      for p in df_valid['Paradigm']]

        # ── Plot 1: Accuracy vs FLOPs (scatter) ──────────────────────────
        ax = axes[0, 0]
        for _, row in df_valid.iterrows():
            c = colors.get(row['Paradigm'], '#6B7280')
            ax.scatter(row['FLOPs_G'], row['Accuracy_%'],
                       s=row['Model_Size_MB']*8, c=c, alpha=0.8, zorder=3)
            ax.annotate(row['Model'], (row['FLOPs_G'], row['Accuracy_%']),
                        fontsize=7.5, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('FLOPs (GFLOPs)', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Accuracy vs Computational Cost\n(bubble size = model size MB)',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── Plot 2: Energy Efficiency Score ──────────────────────────────
        ax = axes[0, 1]
        models = df_valid['Model'].tolist()
        ees    = df_valid['EES'].tolist()
        x      = range(len(models))
        bars   = ax.bar(x, ees, color=bar_colors, alpha=0.85,
                        edgecolor='white', linewidth=1.2, zorder=3)
        for bar, val in zip(bars, ees):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('Energy Efficiency Score (EES)', fontsize=10)
        ax.set_title('Energy Efficiency Score\n(Higher = More Efficient)',
                     fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── Plot 3: Latency vs Accuracy ───────────────────────────────────
        ax = axes[1, 0]
        for _, row in df_valid.iterrows():
            c = colors.get(row['Paradigm'], '#6B7280')
            ax.scatter(row['Latency_Mean_ms'], row['Accuracy_%'],
                       s=120, c=c, alpha=0.8, zorder=3)
            ax.annotate(row['Model'],
                        (row['Latency_Mean_ms'], row['Accuracy_%']),
                        fontsize=7.5, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('Inference Latency (ms)', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Accuracy vs Inference Latency\n(lower-right = ideal)',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── Plot 4: Clinical Deployment Score ────────────────────────────
        ax = axes[1, 1]
        cds = df_valid['CDS'].tolist()
        bars = ax.bar(x, cds, color=bar_colors, alpha=0.85,
                      edgecolor='white', linewidth=1.2, zorder=3)
        for bar, val in zip(bars, cds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('Clinical Deployment Score (CDS)', fontsize=10)
        ax.set_title('Clinical Deployment Score\n(Accuracy + Speed + Portability)',
                     fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        patches = [mpatches.Patch(color=v, label=k)
                   for k, v in colors.items()
                   if k in df_valid['Paradigm'].values]
        fig.legend(handles=patches, loc='lower center', ncol=4,
                   fontsize=9, title='Architecture Paradigm',
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'efficiency_chart.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Charts saved → {chart_path}")

    except ImportError:
        print("  ⚠ matplotlib not found → skipping charts")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT NOVELTY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_novelty_summary(df, output_dir):
    print("\n" + "="*90)
    print("  NOVELTY SUMMARY — HOW THIS EXTENDS SHALABY ET AL.")
    print("="*90)

    print("""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  SHALABY ART.1 (Pose Estimation)     → Spatial joint localization        │
  │  SHALABY ART.3 (GPU Skinning)        → Energy-aware GPU computation      │
  │                                                                           │
  │  THIS STUDY (RE-Bench + Efficiency)  → Extends both:                     │
  │  1. Reproducible multi-paradigm benchmarking (beyond single-paradigm)    │
  │  2. Energy-aware inference analysis (adapts GPU Skinning principles      │
  │     to clinical AI inference domain)                                     │
  │  3. FLOPs + Latency + EES metrics (new for rehabilitation AI)            │
  │  4. Clinical deployment matrix (directly actionable)                     │
  └──────────────────────────────────────────────────────────────────────────┘
    """)

    # Best models per dimension
    df_v = df[df['Accuracy_%'] > 0]
    if len(df_v) > 0:
        best_acc = df_v.loc[df_v['Accuracy_%'].idxmax()]
        best_ees = df_v.loc[df_v['EES'].idxmax()]
        best_lat = df_v.loc[df_v['Latency_Mean_ms'].idxmin()]
        best_sml = df_v.loc[df_v['Model_Size_MB'].idxmin()]

        print(f"  BEST ACCURACY:    {best_acc['Model']} "
              f"({best_acc['Accuracy_%']:.2f}%)")
        print(f"  BEST EFFICIENCY:  {best_ees['Model']} "
              f"(EES={best_ees['EES']:.3f})")
        print(f"  LOWEST LATENCY:   {best_lat['Model']} "
              f"({best_lat['Latency_Mean_ms']:.1f}ms)")
        print(f"  SMALLEST MODEL:   {best_sml['Model']} "
              f"({best_sml['Model_Size_MB']:.1f}MB)")

    # Save novelty summary
    summary = os.path.join(output_dir, 'novelty_efficiency_summary.txt')
    with open(summary, 'w') as f:
        f.write("ENERGY-AWARE EFFICIENCY ANALYSIS — NOVELTY SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("Inspired by: Energy-Aware GPU Skinning (Shalaby et al.)\n")
        f.write("Adapted to: Clinical AI Inference for Rehabilitation\n\n")
        f.write("New Metrics Introduced:\n")
        f.write("  1. FLOPs (GFLOPs) per inference\n")
        f.write("  2. Model Size (MB)\n")
        f.write("  3. CPU Inference Latency (ms)\n")
        f.write("  4. Energy Efficiency Score (EES)\n")
        f.write("  5. Clinical Deployment Score (CDS)\n\n")
        f.write("Results:\n")
        if len(df) > 0:
            f.write(df.to_string(index=False))
    print(f"  ✓ Novelty summary → {summary}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Energy-Aware Efficiency Analysis for RE-Bench')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--n_runs', type=int, default=50,
                        help='Number of inference runs for latency measurement')
    parser.add_argument('--skip_new_models', action='store_true',
                        help='Skip ConvNeXt and EfficientNetV2 (needs timm)')
    args = parser.parse_args()

    if args.skip_new_models:
        del MODEL_BUILDERS['ConvNeXt-Tiny']
        del MODEL_BUILDERS['EfficientNetV2']

    print("\n  Requirements check:")
    try:
        import thop
        print("  ✓ thop available — exact FLOPs will be computed")
    except ImportError:
        print("  ⚠ thop not found — using parameter-based FLOPs estimate")
        print("    Install: pip install thop")
    try:
        import timm
        print("  ✓ timm available — ViT-Tiny, ConvNeXt, EfficientNetV2 included")
    except ImportError:
        print("  ⚠ timm not found — ViT-Tiny, ConvNeXt will be skipped")
        print("    Install: pip install timm")

    # Run
    df = run_efficiency_analysis(args.output_dir, args.n_runs)
    print_efficiency_tables(df, args.output_dir)
    make_efficiency_charts(df, args.output_dir)
    print_novelty_summary(df, args.output_dir)

    print("\n" + "="*78)
    print("  ANALYSIS COMPLETE")
    print(f"  Output dir: {args.output_dir}")
    print("  Files generated:")
    print("    - efficiency_analysis.csv")
    print("    - energy_aware_report.txt")
    print("    - efficiency_chart.png")
    print("    - novelty_efficiency_summary.txt")
    print("="*78)
