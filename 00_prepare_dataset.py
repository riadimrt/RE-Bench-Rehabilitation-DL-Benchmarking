"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  00_prepare_dataset.py                                                       ║
║  SCRIPT PERSIAPAN DATA — otomatis organisir frame ke class_0 ~ class_6      ║
║                                                                              ║
║  Menangani SEMUA skenario:                                                   ║
║    A) Folder flat (semua BMP campur, pakai .mat untuk label)                 ║
║    B) Folder flat (nama file mengandung label)                               ║
║    C) Video .wmv → ekstrak frame → organisir                                 ║
║    D) Folder sudah ada subfolder (verifikasi saja)                           ║
║                                                                              ║
║  Jalankan:                                                                   ║
║    python 00_prepare_dataset.py                          (mode interaktif)  ║
║    python 00_prepare_dataset.py --source "..\\Frame 2"  (langsung)          ║
║    python 00_prepare_dataset.py --source "..\\Frame 2" --mat "..\\ciri_chain_code.mat"
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import shutil
import argparse
import glob
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# KELAS & LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: 'Tidak_Ada_Objek',
    1: 'Tdk_Dikenal',
    2: 'Ekstensi',
    3: 'Fleksi',
    4: 'Abduksi',
    5: 'Hiperekstensi',
    6: 'Adduksi',
}
NUM_CLASSES = 7

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: cek ekstensi gambar
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def is_image(path):
    return Path(path).suffix.lower() in IMG_EXTS

def find_all_images(folder):
    """Temukan semua file gambar dalam folder (tidak rekursif)."""
    images = []
    for f in sorted(Path(folder).iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            images.append(f)
    return images

def find_all_images_recursive(folder):
    """Temukan semua file gambar secara rekursif."""
    images = []
    for ext in IMG_EXTS:
        images.extend(Path(folder).rglob(f'*{ext}'))
        images.extend(Path(folder).rglob(f'*{ext.upper()}'))
    return sorted(set(images))


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO A: Pakai file .mat untuk label
# ─────────────────────────────────────────────────────────────────────────────
def load_labels_from_mat(mat_path):
    """
    Baca label dari ciri_chain_code.mat atau ciri_moment_invariant.mat.
    Kolom terakhir adalah label kelas.
    Returns: list of int labels
    """
    try:
        import scipy.io as sio
        mat = sio.loadmat(mat_path)
        
        # Cari key yang relevan
        data_key = None
        for k in mat.keys():
            if not k.startswith('_') and hasattr(mat[k], 'shape'):
                if len(mat[k].shape) == 2 and mat[k].shape[1] >= 2:
                    data_key = k
                    break
        
        if data_key is None:
            print(f"  ⚠ Tidak bisa membaca key dari {mat_path}")
            return None
        
        data = mat[data_key]
        
        # Skip baris pertama jika header (NaN atau string)
        try:
            labels_raw = data[1:, -1]  # baris 2 dst, kolom terakhir
        except:
            labels_raw = data[:, -1]
        
        labels = [int(round(float(l))) for l in labels_raw]
        print(f"  ✓ Berhasil baca {len(labels)} label dari {mat_path}")
        print(f"  Distribusi: {Counter(labels)}")
        return labels
        
    except ImportError:
        print("  ⚠ scipy tidak tersedia. Install: pip install scipy")
        return None
    except Exception as e:
        print(f"  ⚠ Error baca .mat: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO B: Parse label dari nama file
# ─────────────────────────────────────────────────────────────────────────────
def parse_label_from_filename(filename):
    """
    Coba parse label kelas dari nama file.
    Contoh: 'ekstensi_001.bmp' → 2
             'class2_frame001.bmp' → 2
             'frame_003_label_4.bmp' → 4
    """
    name = filename.lower()
    
    # Cek keyword kelas
    keyword_map = [
        (['tidak_ada', 'tidakada', 'no_obj', 'noobj', 'class0', 'class_0', '_0_'], 0),
        (['tdk_dikenal', 'tdkdikenal', 'unknown', 'tidak_dikenal', 'class1', 'class_1', '_1_'], 1),
        (['hiperekstensi', 'hyperext', 'hyper'], 5),   # harus sebelum 'ekstensi'
        (['ekstensi', 'extension', 'class2', 'class_2', '_2_'], 2),
        (['fleksi', 'flexion', 'flex', 'class3', 'class_3', '_3_'], 3),
        (['abduksi', 'abduction', 'abduct', 'class4', 'class_4', '_4_'], 4),
        (['adduksi', 'adduction', 'adduct', 'class6', 'class_6', '_6_'], 6),
    ]
    
    for keywords, label in keyword_map:
        for kw in keywords:
            if kw in name:
                return label
    
    # Coba cari pola 'label_X' atau '_lX_' dalam nama
    import re
    m = re.search(r'label[_\s]?(\d)', name)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 6:
            return val
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO C: Ekstrak frame dari video .wmv
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames_from_video(video_path, output_dir, prefix='frame'):
    """
    Ekstrak semua frame dari video menggunakan OpenCV.
    """
    try:
        import cv2
    except ImportError:
        print("  ⚠ OpenCV tidak tersedia. Install: pip install opencv-python")
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠ Tidak bisa membuka video: {video_path}")
        return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {video_path.name}  |  {total} frames  |  {fps:.1f} fps")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}.bmp")
        cv2.imwrite(out_path, frame)
        saved_paths.append(out_path)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"    Ekstrak frame {frame_idx}/{total}...", end='\r')
    
    cap.release()
    print(f"\n  ✓ Tersimpan {frame_idx} frame ke {output_dir}")
    return saved_paths


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Organisir gambar ke folder class_0 ~ class_6
# ─────────────────────────────────────────────────────────────────────────────
def organize_to_class_folders(image_paths, labels, output_base, copy=True, force=False):
    """
    Copy/move gambar ke:
        output_base/class_0/
        output_base/class_1/
        ...
        output_base/class_6/
    
    image_paths : list of Path
    labels      : list of int (sama panjang dengan image_paths)
    copy        : True=copy, False=move
    """
    assert len(image_paths) == len(labels), \
        f"Jumlah gambar ({len(image_paths)}) ≠ jumlah label ({len(labels)})"
    
    # Buat semua subfolder
    for c in range(NUM_CLASSES):
        os.makedirs(os.path.join(output_base, f'class_{c}'), exist_ok=True)
    
    counts = Counter()
    errors = []
    
    for img_path, label in zip(image_paths, labels):
        if label < 0 or label >= NUM_CLASSES:
            errors.append(f"Label {label} out of range: {img_path.name}")
            continue
        
        dst_folder = os.path.join(output_base, f'class_{label}')
        dst_path   = os.path.join(dst_folder, img_path.name)
        
        # Hindari overwrite: tambah suffix jika perlu
        if os.path.exists(dst_path) and not force:
            stem   = img_path.stem
            suffix = img_path.suffix
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_folder, f"{stem}_{counter}{suffix}")
                counter += 1
        
        try:
            if copy:
                shutil.copy2(str(img_path), dst_path)
            else:
                shutil.move(str(img_path), dst_path)
            counts[label] += 1
        except Exception as e:
            errors.append(f"Error {img_path.name}: {e}")
    
    return counts, errors


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY: Cek dataset yang sudah ada
# ─────────────────────────────────────────────────────────────────────────────
def verify_dataset(data_dir):
    """Verifikasi struktur dan isi folder data."""
    data_dir = Path(data_dir)
    print(f"\n  {'='*60}")
    print(f"  VERIFIKASI DATASET: {data_dir}")
    print(f"  {'='*60}")
    
    total = 0
    missing = []
    
    for c in range(NUM_CLASSES):
        folder = data_dir / f'class_{c}'
        if not folder.exists():
            missing.append(c)
            print(f"  ❌ class_{c} ({CLASS_NAMES[c]:20s}): FOLDER TIDAK ADA")
            continue
        
        imgs = find_all_images(folder)
        n = len(imgs)
        total += n
        
        status = '✓' if n > 0 else '⚠'
        bar    = '█' * min(30, n // 5) if n > 0 else ''
        print(f"  {status} class_{c} ({CLASS_NAMES[c]:20s}): {n:4d} gambar  {bar}")
    
    print(f"  {'─'*60}")
    print(f"  TOTAL: {total} gambar")
    
    if missing:
        print(f"\n  ⚠ Folder tidak ada: {missing}")
        print("    Jalankan script ini untuk membuatnya.")
        return False
    elif total == 0:
        print("\n  ⚠ Semua folder kosong!")
        return False
    else:
        print(f"\n  ✓ Dataset siap digunakan!")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Persiapan dataset untuk CNN/LSTM modern',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Mode interaktif (ditanya satu per satu):
  python 00_prepare_dataset.py

  # Dari folder flat dengan file .mat untuk label:
  python 00_prepare_dataset.py --source "..\\Frame 2" --mat "..\\ciri_chain_code.mat" --dest .\\data

  # Dari folder flat, parse label dari nama file:
  python 00_prepare_dataset.py --source "..\\Frame 2" --dest .\\data

  # Dari video .wmv (langsung extract + organisir):
  python 00_prepare_dataset.py --video "..\\video.wmv" --mat "..\\ciri_chain_code.mat" --dest .\\data

  # Hanya verifikasi dataset yang sudah ada:
  python 00_prepare_dataset.py --verify_only --dest .\\data
        """
    )
    parser.add_argument('--source',      type=str, default=None,
                        help='Folder sumber berisi gambar (flat atau bersubfolder)')
    parser.add_argument('--video',       type=str, default=None,
                        help='File video .wmv untuk diekstrak framenya')
    parser.add_argument('--mat',         type=str, default=None,
                        help='File .mat berisi label (ciri_chain_code.mat atau ciri_moment_invariant.mat)')
    parser.add_argument('--dest',        type=str, default='./data',
                        help='Folder tujuan (default: ./data)')
    parser.add_argument('--move',        action='store_true',
                        help='Pindahkan file (default: copy)')
    parser.add_argument('--verify_only', action='store_true',
                        help='Hanya verifikasi, tidak mengubah apa-apa')
    parser.add_argument('--force',       action='store_true',
                        help='Timpa file yang sudah ada')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  PERSIAPAN DATASET — Organisir Frame ke Folder Kelas")
    print("="*70)

    dest_dir = Path(args.dest).resolve()

    # ── Hanya verifikasi ─────────────────────────────────────────────────
    if args.verify_only:
        verify_dataset(dest_dir)
        return

    # ── Mode interaktif jika tidak ada argumen ──────────────────────────
    if not args.source and not args.video:
        print("\n  [MODE INTERAKTIF]")
        print("  Tidak ada argumen --source atau --video.")
        print()
        
        print("  Pilih skenario:")
        print("  1. Folder 'Frame 2' berisi gambar flat (semua kelas campur)")
        print("  2. File video .wmv → ekstrak frame dulu")
        print("  3. Dataset sudah terorganisir (hanya verifikasi)")
        print()
        choice = input("  Pilihan (1/2/3): ").strip()
        
        if choice == '3':
            verify_dataset(dest_dir)
            return
        
        if choice == '2':
            video_path = input("  Path ke file .wmv: ").strip().strip('"')
            args.video = video_path
        else:
            source_path = input("  Path ke folder sumber (contoh: ..\\Frame 2): ").strip().strip('"')
            args.source = source_path
        
        mat_path = input("  Path ke file .mat (ciri_chain_code.mat) atau ENTER untuk skip: ").strip().strip('"')
        if mat_path:
            args.mat = mat_path
        
        print(f"  Folder tujuan: {dest_dir}  (ENTER untuk setuju, atau ketik path baru): ", end='')
        new_dest = input().strip().strip('"')
        if new_dest:
            dest_dir = Path(new_dest).resolve()
            args.dest = str(dest_dir)

    print(f"\n  Folder tujuan : {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # STEP 1: Dapatkan daftar gambar
    # ────────────────────────────────────────────────────────────────────
    image_paths = []

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"  ❌ File video tidak ditemukan: {video_path}")
            sys.exit(1)
        
        print(f"\n  [STEP 1] Ekstrak frame dari video...")
        extracted_dir = dest_dir / '_extracted_frames'
        image_paths_str = extract_frames_from_video(
            video_path, str(extracted_dir), prefix='frame'
        )
        image_paths = [Path(p) for p in image_paths_str]
        
    elif args.source:
        source_dir = Path(args.source)
        if not source_dir.exists():
            print(f"  ❌ Folder sumber tidak ditemukan: {source_dir}")
            print(f"     Path absolut yang dicoba: {source_dir.resolve()}")
            sys.exit(1)
        
        print(f"\n  [STEP 1] Membaca gambar dari: {source_dir}")
        
        # Cek apakah sudah ada subfolder kelas
        subfolders = [d for d in source_dir.iterdir() if d.is_dir()]
        class_folders = [d for d in subfolders if any(
            d.name.lower().startswith(k) for k in
            ['class', '0','1','2','3','4','5','6',
             'ekstensi','fleksi','abduksi','adduksi',
             'hiperekstensi','tidak','tdk','unknown']
        )]
        
        if class_folders:
            print(f"  Ditemukan {len(class_folders)} subfolder kelas!")
            print("  Ini seperti sudah terorganisir. Cek isi...")
            
            # Cek apakah subfolder sudah menggunakan format class_X
            proper_folders = [d for d in source_dir.iterdir()
                             if d.is_dir() and d.name.startswith('class_')]
            
            if proper_folders and str(source_dir.resolve()) == str(dest_dir.resolve()):
                print("  ✓ Folder sudah dalam format class_X — langsung verifikasi.")
                verify_dataset(source_dir)
                return
            
            # Copy subfolder ke dest dengan format yang benar
            _copy_subfolders(source_dir, dest_dir)
            verify_dataset(dest_dir)
            return
        
        # Flat directory
        image_paths = find_all_images(source_dir)
        if not image_paths:
            image_paths = find_all_images_recursive(source_dir)
        
        print(f"  Ditemukan {len(image_paths)} gambar")

    if not image_paths:
        print("  ❌ Tidak ada gambar ditemukan!")
        sys.exit(1)

    # ────────────────────────────────────────────────────────────────────
    # STEP 2: Dapatkan label
    # ────────────────────────────────────────────────────────────────────
    print(f"\n  [STEP 2] Menentukan label untuk {len(image_paths)} gambar...")
    labels = None

    # Prioritas 1: File .mat
    if args.mat:
        mat_path = Path(args.mat)
        if not mat_path.exists():
            print(f"  ⚠ File .mat tidak ditemukan: {mat_path}")
        else:
            labels_from_mat = load_labels_from_mat(str(mat_path))
            if labels_from_mat and len(labels_from_mat) == len(image_paths):
                labels = labels_from_mat
                print(f"  ✓ Label dari .mat: {len(labels)} label cocok dengan {len(image_paths)} gambar")
            elif labels_from_mat:
                print(f"  ⚠ Jumlah label dari .mat ({len(labels_from_mat)}) ≠ jumlah gambar ({len(image_paths)})")
                print("  Mencoba mencocokkan berdasarkan urutan...")
                
                if len(labels_from_mat) < len(image_paths):
                    # Ambil sebagian gambar sesuai jumlah label
                    print(f"  Menggunakan {len(labels_from_mat)} gambar pertama")
                    image_paths = image_paths[:len(labels_from_mat)]
                    labels = labels_from_mat
                else:
                    # Ambil sebagian label
                    labels = labels_from_mat[:len(image_paths)]
                    print(f"  Menggunakan {len(labels)} label pertama")

    # Prioritas 2: Parse dari nama file
    if labels is None:
        print("  Mencoba parse label dari nama file...")
        labels_from_name = []
        unresolved = []
        
        for img_path in image_paths:
            lbl = parse_label_from_filename(img_path.name)
            labels_from_name.append(lbl)
            if lbl is None:
                unresolved.append(img_path.name)
        
        resolved = [l for l in labels_from_name if l is not None]
        print(f"  Berhasil parse {len(resolved)}/{len(image_paths)} label dari nama file")
        
        if unresolved[:5]:
            print(f"  Contoh tidak terparse: {unresolved[:5]}")
        
        if len(resolved) >= len(image_paths) * 0.8:
            # Lebih dari 80% berhasil, gunakan ini
            # Untuk yang None, coba tanya atau assign ke kelas default
            labels = []
            for l in labels_from_name:
                labels.append(l if l is not None else 0)
        elif len(resolved) > 0:
            print(f"  Hanya {len(resolved)} gambar bisa diparse — skip yang tidak terparse")
            paired = [(p, l) for p, l in zip(image_paths, labels_from_name) if l is not None]
            image_paths = [x[0] for x in paired]
            labels      = [x[1] for x in paired]

    # Prioritas 3: Manual assignment (mode interaktif)
    if labels is None or len(labels) == 0:
        print("\n  ⚠ Tidak bisa menentukan label otomatis.")
        print("  Gunakan salah satu cara:")
        print("    1. Tambahkan argumen --mat <path_ke_ciri_chain_code.mat>")
        print("    2. Ganti nama file dengan keyword kelas (ekstensi_, fleksi_, dll.)")
        print("    3. Organisir gambar manual ke subfolder class_0 ~ class_6")
        print()
        print("  Atau lanjutkan dengan PEMBAGIAN MERATA (untuk testing saja)?")
        ans = input("  Bagi merata ke 7 kelas untuk testing? (y/n): ").strip().lower()
        if ans == 'y':
            n = len(image_paths)
            labels = [i % NUM_CLASSES for i in range(n)]
            print(f"  ⚠ PERINGATAN: Label dibagi MERATA secara dummy — hanya untuk testing!")
        else:
            print("  Batalkan. Tidak ada perubahan.")
            sys.exit(0)

    # ────────────────────────────────────────────────────────────────────
    # STEP 3: Copy/move ke folder kelas
    # ────────────────────────────────────────────────────────────────────
    action = 'PINDAHKAN' if args.move else 'COPY'
    print(f"\n  [STEP 3] {action} {len(image_paths)} gambar ke {dest_dir}...")
    print(f"  Distribusi label yang akan diorganisir:")
    c = Counter(labels)
    for cls_id in sorted(c.keys()):
        print(f"    class_{cls_id} ({CLASS_NAMES.get(cls_id,'?'):20s}): {c[cls_id]} gambar")

    # Konfirmasi
    confirm = input(f"\n  Lanjutkan? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Dibatalkan.")
        sys.exit(0)

    counts, errors = organize_to_class_folders(
        image_paths, labels, str(dest_dir),
        copy=not args.move, force=args.force
    )

    print(f"\n  {'='*60}")
    print(f"  SELESAI! Hasil organisir:")
    for cls_id in range(NUM_CLASSES):
        n = counts.get(cls_id, 0)
        bar = '█' * min(30, n // 5) if n > 0 else '(kosong)'
        print(f"  class_{cls_id} ({CLASS_NAMES[cls_id]:20s}): {n:4d}  {bar}")

    if errors:
        print(f"\n  ⚠ {len(errors)} error:")
        for e in errors[:10]:
            print(f"    {e}")

    # Final verification
    verify_dataset(dest_dir)

    print(f"\n  ✓ Dataset siap! Sekarang jalankan:")
    print(f"    python 01_cnn_transfer_learning.py --data_dir {dest_dir} \\")
    print(f"           --models mobilenet efficientnet \\")
    print(f"           --epochs_warmup 5 --epochs_finetune 25 \\")
    print(f"           --output_dir ./output")


def _copy_subfolders(source_dir, dest_dir):
    """Copy isi subfolder kelas ke format class_0 ~ class_6."""
    label_map = {
        '0':0,'class_0':0,'class0':0,'tidakada':0,'tidak_ada':0,'no_object':0,
        '1':1,'class_1':1,'class1':1,'unknown':1,'tdk_dikenal':1,'tdkdikenal':1,
        '2':2,'class_2':2,'class2':2,'ekstensi':2,'extension':2,
        '3':3,'class_3':3,'class3':3,'fleksi':3,'flexion':3,
        '4':4,'class_4':4,'class4':4,'abduksi':4,'abduction':4,
        '5':5,'class_5':5,'class5':5,'hiperekstensi':5,'hyperextension':5,
        '6':6,'class_6':6,'class6':6,'adduksi':6,'adduction':6,
    }

    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir():
            continue
        label = label_map.get(folder.name.lower())
        if label is None:
            print(f"  ⚠ Tidak bisa map folder '{folder.name}' → skip")
            continue

        dst = dest_dir / f'class_{label}'
        os.makedirs(dst, exist_ok=True)
        imgs = find_all_images(folder)
        for img in imgs:
            shutil.copy2(str(img), str(dst / img.name))
        print(f"  ✓ {folder.name} → class_{label}: {len(imgs)} gambar")


if __name__ == '__main__':
    main()
