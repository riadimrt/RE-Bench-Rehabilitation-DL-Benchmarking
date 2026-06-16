# Rekomendasi Arsitektur Modern untuk Paper

## Dataset: 863 frame, 7 kelas, single-subject

### PERINGKAT REKOMENDASI:

| Rank | Model | Acc. Estimasi | GPU? | Waktu Train | Cocok Paper? |
|------|-------|--------------|------|-------------|--------------|
| 1 | MobileNetV2 + Transfer Learning | ~92-95% | Tidak wajib | ~5 menit | ★★★★★ |
| 2 | EfficientNet-B0 + TL | ~91-94% | Tidak wajib | ~8 menit | ★★★★★ |
| 3 | MediaPipe Skeleton + LSTM | ~85-90% | Tidak perlu | ~3 menit | ★★★★☆ |
| 4 | ResNet18 + Transfer Learning | ~90-93% | Dianjurkan | ~6 menit | ★★★★☆ |
| 5 | Skeleton + ST-GCN (ringan) | ~82-88% | Dianjurkan | ~10 menit | ★★★☆☆ |
| 6 | Vision Transformer (ViT-tiny) | ~85-90% | Wajib GPU | ~20 menit | ★★☆☆☆ |

### REKOMENDASI UNTUK PAPER:
Gunakan **MobileNetV2 + EfficientNet-B0 + MediaPipe+LSTM** sebagai 3 representasi modern:
- CNN (spatial feature, transfer learning)
- Skeleton+Temporal (pose-based, LSTM)
- Lightweight Transformer (opsional)

Semua dievaluasi dengan S5-CV untuk konsistensi dengan baseline klasik.
