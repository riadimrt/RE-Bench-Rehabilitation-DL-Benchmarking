"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODULE 4: MASTER COMPARISON SCRIPT                                         ║
║  Runs ALL models and produces final comparison table                        ║
║  Classical (from hasil_cv.csv) + Modern (CNN, LSTM, ViT, GCN)              ║
║                                                                              ║
║  Usage:                                                                      ║
║    python 04_master_comparison.py --data_dir PATH --classical_csv PATH      ║
║    python 04_master_comparison.py --only_compare  (skip training, plot only)║
╚══════════════════════════════════════════════════════════════════════════════╝

Output:
    - hasil_comparison_full.csv  : all models side-by-side
    - comparison_chart.png       : bar chart for paper figure
    - comparison_table.txt       : formatted ASCII table for paper
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_classical_results(csv_path):
    """Load classical ML baseline from existing hasil_cv.csv."""
    if not os.path.exists(csv_path):
        print(f"  ⚠ Classical results not found: {csv_path}")
        print("    Using placeholder values from paper (Python S5-CV results)")
        return pd.DataFrame({
            'Model':            ['Decision Tree','BPNN','RBFNN','Naïve Bayes','Random Forest','SVM'],
            'Feature':          ['Chain Code']*6,
            'Accuracy_Mean(%)': [81.12, 84.36, 62.92, 78.45, 88.07, 87.25],
            'Accuracy_Std(%)':  [4.00,  3.67,  2.81,  1.55,  2.09,  2.65],
            'F1_Score(%)':      [80.49, 84.35, 62.37, 76.92, 86.41, 85.54],
            'Category':         ['Classical']*6,
        })

    df = pd.read_csv(csv_path)
    df['Feature']  = 'Chain Code'
    df['Category'] = 'Classical'
    return df


def make_comparison_chart(df_combined, output_dir='.'):
    """Create paper-ready bar chart comparing all models."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            'Comparative Evaluation: Classical vs. Modern ML Algorithms\n'
            'Human Body Motion Classification (S5-CV, n=863, 7 classes)',
            fontsize=14, fontweight='bold', y=1.01
        )

        colors = {
            'Classical':  '#2563EB',
            'CNN':        '#16A34A',
            'Skeleton':   '#D97706',
            'Transformer':'#9333EA',
            'GCN':        '#DC2626',
        }

        for ax_idx, metric in enumerate(['Accuracy_Mean(%)', 'F1_Score(%)']):
            ax = axes[ax_idx]
            title = 'Accuracy (%)' if metric == 'Accuracy_Mean(%)' else 'F1-Score (Macro, %)'

            models = df_combined['Model'].tolist()
            values = df_combined[metric].astype(float).tolist()
            stds   = df_combined.get('Accuracy_Std(%)', pd.Series([0]*len(df_combined))).astype(float).tolist()
            cats   = df_combined['Category'].tolist()

            bar_colors = [colors.get(c, '#6B7280') for c in cats]
            x = np.arange(len(models))

            bars = ax.bar(x, values, color=bar_colors, alpha=0.85,
                          yerr=stds if metric=='Accuracy_Mean(%)' else None,
                          capsize=4, edgecolor='white', linewidth=1.2, zorder=3)

            # Value labels on bars
            for bar, val, std in zip(bars, values, stds):
                h = bar.get_height()
                label = f'{val:.1f}' + (f'\n±{std:.1f}' if metric=='Accuracy_Mean(%)' and std > 0 else '')
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        label, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
            ax.set_ylabel(title, fontsize=11)
            ax.set_ylim(0, 105)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Horizontal reference lines
            for y_val, label, style in [(88.07,'RF (Classical Best)','--'), (90,'Target ~90%',':')]:
                ax.axhline(y=y_val, color='gray', linestyle=style, alpha=0.5, linewidth=1)
                ax.text(len(models)-0.5, y_val+0.3, label, ha='right', va='bottom',
                        fontsize=7, color='gray')

        # Legend
        patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()
                   if k in df_combined['Category'].values]
        fig.legend(handles=patches, loc='upper right', fontsize=10,
                   title='Algorithm Category', title_fontsize=10,
                   bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()
        out_path = os.path.join(output_dir, 'comparison_chart.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Chart saved → {out_path}")
        return out_path

    except ImportError:
        print("  ⚠ matplotlib not installed — skipping chart (pip install matplotlib)")
        return None


def print_latex_table(df):
    """Print LaTeX table for direct copy-paste into paper."""
    print("\n" + "="*78)
    print("  LATEX TABLE (copy into paper)")
    print("="*78)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Comprehensive comparison of classical and modern ML algorithms}")
    print(r"\label{tab:full_comparison}")
    print(r"\begin{tabular}{llccccc}")
    print(r"\hline")
    print(r"\textbf{Category} & \textbf{Model} & \textbf{Feature} & "
          r"\textbf{Acc (\%)} & \textbf{Std} & \textbf{F1 (\%)} & \textbf{Prec (\%)} \\")
    print(r"\hline")

    prev_cat = None
    for _, row in df.iterrows():
        cat = row.get('Category', '')
        if cat != prev_cat:
            if prev_cat is not None:
                print(r"\hline")
            prev_cat = cat

        model  = str(row.get('Model', ''))
        feat   = str(row.get('Feature', '-'))
        acc    = float(row.get('Accuracy_Mean(%)', 0))
        std    = float(row.get('Accuracy_Std(%)', 0))
        f1     = float(row.get('F1_Score(%)', 0))
        prec   = float(row.get('Precision(%)', 0))

        bold_start = r"\textbf{" if acc >= 88.0 else ""
        bold_end   = "}"         if acc >= 88.0 else ""

        print(f"{cat} & {bold_start}{model}{bold_end} & {feat} & "
              f"{bold_start}{acc:.2f}{bold_end} & {std:.2f} & "
              f"{f1:.2f} & {prec:.2f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_ascii_summary(df):
    """Print formatted ASCII comparison table."""
    print("\n" + "═"*100)
    print("  COMPREHENSIVE COMPARISON — ALL ALGORITHMS (Stratified 5-Fold CV)")
    print("═"*100)
    print(f"  {'Category':12s} {'Model':22s} {'Feature':12s} "
          f"{'Acc(%)':>10s} {'±Std':>6s} {'F1(%)':>8s} {'Prec(%)':>9s} {'Note':>10s}")
    print(f"  {'─'*12} {'─'*22} {'─'*12} {'─'*10} {'─'*6} {'─'*8} {'─'*9} {'─'*10}")

    best_acc = df['Accuracy_Mean(%)'].astype(float).max()
    for _, row in df.iterrows():
        acc  = float(row.get('Accuracy_Mean(%)', 0))
        std  = float(row.get('Accuracy_Std(%)', 0))
        f1   = float(row.get('F1_Score(%)', 0))
        prec = float(row.get('Precision(%)', 0))
        star = ' ★ BEST' if acc == best_acc else ''

        print(f"  {str(row.get('Category','')): <12s} "
              f"{str(row.get('Model','')):<22s} "
              f"{str(row.get('Feature','-')):<12s} "
              f"{acc:>10.2f} {std:>6.2f} {f1:>8.2f} {prec:>9.2f}{star}")

    print("═"*100)

    # Category summary
    print("\n  CATEGORY SUMMARY (mean across models):")
    for cat, grp in df.groupby('Category'):
        mean_acc = grp['Accuracy_Mean(%)'].astype(float).mean()
        best_in_cat = grp['Accuracy_Mean(%)'].astype(float).max()
        print(f"  {cat:15s}: mean={mean_acc:.2f}%  best={best_in_cat:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_all(data_dir, classical_csv, run_cnn=True, run_lstm=True,
            run_vit=False, run_gcn=False, output_dir='./output',
            only_compare=False):
    """
    Orchestrate all training and produce final comparison.
    Set run_vit=False by default (needs GPU + timm for good results).
    """
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    # ── Load classical baseline ────────────────────────────────────────────
    print("\n  Loading classical ML results...")
    df_classical = load_classical_results(classical_csv)

    # Best per-feature for classical
    for _, row in df_classical.iterrows():
        all_rows.append({
            'Category': 'Classical',
            'Model': row.get('Algoritma', row.get('Model', 'Unknown')),
            'Feature': 'Chain Code',
            'Accuracy_Mean(%)': float(str(row.get('CC Acc Mean(%)', row.get('Accuracy_Mean(%)', 0))).replace('%','')),
            'Accuracy_Std(%)':  float(str(row.get('CC Acc Std(%)',  row.get('Accuracy_Std(%)', 0))).replace('%','')),
            'F1_Score(%)':      float(str(row.get('CC F1(%)',        row.get('F1_Score(%)', 0))).replace('%','')),
            'Precision(%)':     float(str(row.get('CC Precision(%)', row.get('Precision(%)', 0))).replace('%','')),
            'Recall(%)':        float(str(row.get('CC Recall(%)',    row.get('Recall(%)', 0))).replace('%','')),
        })

    if only_compare:
        print("  --only_compare mode: loading existing CSV results...")
        for fname, cat in [('hasil_cnn_cv.csv','CNN'),
                           ('hasil_skeleton_lstm_cv.csv','Skeleton'),
                           ('hasil_transformer_gcn_cv.csv','Transformer/GCN')]:
            p = os.path.join(output_dir, fname)
            if os.path.exists(p):
                df = pd.read_csv(p)
                df['Category'] = cat
                df['Feature']  = 'RGB' if cat=='CNN' else ('Skeleton' if cat=='Skeleton' else 'Image/Graph')
                all_rows.extend(df.to_dict('records'))
    else:
        # ── CNN Transfer Learning ──────────────────────────────────────────
        if run_cnn:
            print("\n" + "─"*60)
            print("  RUNNING: CNN Transfer Learning...")
            try:
                from importlib import import_module
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                m = import_module('01_cnn_transfer_learning')
                cnn_results = m.run_evaluation(
                    data_dir=data_dir,
                    model_names=['mobilenet', 'efficientnet'],
                    epochs_warmup=5, epochs_finetune=20,
                    output_dir=output_dir
                )
                for r in cnn_results:
                    all_rows.append({
                        'Category': 'CNN', 'Feature': 'RGB (Transfer)',
                        'Model': r['model'], 'Accuracy_Mean(%)': r['mean_acc'],
                        'Accuracy_Std(%)': r['std_acc'], 'F1_Score(%)': r['mean_f1'],
                        'Precision(%)': r['precision'], 'Recall(%)': r['recall'],
                    })
            except Exception as e:
                print(f"  ⚠ CNN training error: {e}")
                print("    Run manually: python 01_cnn_transfer_learning.py --data_dir PATH")

        # ── Skeleton + LSTM ────────────────────────────────────────────────
        if run_lstm:
            print("\n" + "─"*60)
            print("  RUNNING: Skeleton + LSTM...")
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                m2 = import_module('02_skeleton_lstm')
                lstm_results = m2.run_skeleton_evaluation(
                    data_dir=data_dir,
                    model_names=['lstm', 'bilstm', 'attn_lstm'],
                    seq_len=1,
                    output_dir=output_dir
                )
                for r in lstm_results:
                    all_rows.append({
                        'Category': 'Skeleton+RNN', 'Feature': 'MediaPipe (132-dim)',
                        'Model': f"Skel+{r['model'].upper()}", 'Accuracy_Mean(%)': r['mean_acc'],
                        'Accuracy_Std(%)': r['std_acc'], 'F1_Score(%)': r['mean_f1'],
                        'Precision(%)': r['precision'], 'Recall(%)': r['recall'],
                    })
            except Exception as e:
                print(f"  ⚠ LSTM training error: {e}")

        # ── ViT + GCN ─────────────────────────────────────────────────────
        if run_vit or run_gcn:
            print("\n" + "─"*60)
            print("  RUNNING: ViT / GCN...")
            try:
                m3 = import_module('03_transformer_gcn')
                tg_results = m3.run_transformer_gcn(
                    data_dir=data_dir, run_vit=run_vit, run_gcn=run_gcn,
                    output_dir=output_dir
                )
                for r in tg_results:
                    cat = 'Transformer' if 'ViT' in r['model'] else 'GCN'
                    all_rows.append({
                        'Category': cat, 'Feature': 'Image patches' if cat=='Transformer' else 'Skeleton graph',
                        'Model': r['model'], 'Accuracy_Mean(%)': r['mean_acc'],
                        'Accuracy_Std(%)': r['std_acc'], 'F1_Score(%)': r['mean_f1'],
                        'Precision(%)': r['precision'], 'Recall(%)': r['recall'],
                    })
            except Exception as e:
                print(f"  ⚠ ViT/GCN error: {e}")

    # ── Combine & Save ─────────────────────────────────────────────────────
    df_final = pd.DataFrame(all_rows)
    out_csv  = os.path.join(output_dir, 'hasil_comparison_full.csv')
    df_final.to_csv(out_csv, index=False)
    print(f"\n  ✓ Full comparison saved → {out_csv}")

    # ── Print summaries ────────────────────────────────────────────────────
    print_ascii_summary(df_final)
    print_latex_table(df_final)
    make_comparison_chart(df_final, output_dir)

    return df_final


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Master comparison: Classical + Modern ML')
    parser.add_argument('--data_dir',      type=str, default='.',
                        help='Directory with frame images in class subfolders')
    parser.add_argument('--classical_csv', type=str, default='./hasil_cv.csv',
                        help='Path to classical ML CV results CSV')
    parser.add_argument('--output_dir',    type=str, default='./output_modern')
    parser.add_argument('--run_cnn',       action='store_true', default=True)
    parser.add_argument('--run_lstm',      action='store_true', default=True)
    parser.add_argument('--run_vit',       action='store_true', default=False,
                        help='Enable ViT (needs GPU + timm)')
    parser.add_argument('--run_gcn',       action='store_true', default=False,
                        help='Enable GCN (needs MediaPipe)')
    parser.add_argument('--only_compare',  action='store_true', default=False,
                        help='Skip training, combine existing CSV outputs only')
    args = parser.parse_args()

    df = run_all(
        data_dir=args.data_dir,
        classical_csv=args.classical_csv,
        run_cnn=args.run_cnn,
        run_lstm=args.run_lstm,
        run_vit=args.run_vit,
        run_gcn=args.run_gcn,
        output_dir=args.output_dir,
        only_compare=args.only_compare,
    )
