"""
Figura 6.14 da Tese:
"Comparison of RMSE and R2 across all six study regions between In-Domain
(Stratified 5-Fold Cross-Validation, left) and Out-of-Domain (Leave-One-Region-Out, right),
demonstrating the domain shift penalty induced by zero-shot spatial transfer without local calibration."
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

OUTPUT_INTERNAL = BASE_DIR / "outputs" / "loro_vs_stratified_comparison.png"
OUTPUT_THESIS = Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results\loro_vs_stratified_comparison.png")

def gerar_figura():
    regions = ['Mafra', 'Monsanto', 'Pombal', 'Serra da Lousã', 'Sintra-Cascais', 'Vila Pouca de Aguiar']
    
    # Métricas retiradas das Tabelas 6.4 e 6.5 da Tese
    strat_rmse = [4.95, 5.14, 5.09, 5.49, 5.26, 4.78]
    loro_rmse  = [5.26, 6.34, 6.24, 7.07, 7.52, 5.34]
    strat_r2   = [0.345, 0.340, 0.297, 0.521, 0.570, 0.373]
    loro_r2    = [0.258, -0.003, -0.056, 0.207, 0.118, 0.218]

    x = np.arange(len(regions))
    width = 0.36

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    c_in = '#1f4e79'    # Deep Navy Blue (In-Domain)
    c_out = '#d35400'   # Vibrant Rust / Terracotta Orange (Out-of-Domain LORO)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 12.5,
        'ytick.labelsize': 13,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 6.2))

    # --- Painel 1: RMSE ---
    rects1 = ax1.bar(x - width/2, strat_rmse, width, label='In-Domain (Stratified 5-Fold CV)',
                     color=c_in, edgecolor='black', linewidth=0.9, zorder=3)
    rects2 = ax1.bar(x + width/2, loro_rmse, width, label='Out-of-Domain (Leave-One-Region-Out)',
                     color=c_out, edgecolor='black', linewidth=0.9, zorder=3)
    ax1.set_ylabel('RMSE (m)', fontweight='bold', labelpad=10)
    ax1.set_title('Root Mean Squared Error (RMSE)', fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=18, ha='right')
    ax1.set_ylim(0, 8.8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f} m', xy=(rect.get_x() + rect.get_width() / 2, h - 0.35), xytext=(0, 0),
                     textcoords='offset points', ha='center', va='top', rotation=90,
                     fontsize=12, fontweight='bold', color='white')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f} m', xy=(rect.get_x() + rect.get_width() / 2, h - 0.35), xytext=(0, 0),
                     textcoords='offset points', ha='center', va='top', rotation=90,
                     fontsize=12, fontweight='bold', color='white')

    # --- Painel 2: R^2 ---
    rects3 = ax2.bar(x - width/2, strat_r2, width, label='In-Domain (Stratified 5-Fold CV)',
                     color=c_in, edgecolor='black', linewidth=0.9, zorder=3)
    rects4 = ax2.bar(x + width/2, loro_r2, width, label='Out-of-Domain (Leave-One-Region-Out)',
                     color=c_out, edgecolor='black', linewidth=0.9, zorder=3)
    ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', fontweight='bold', labelpad=10)
    ax2.set_title(r'Coefficient of Determination ($R^2$)', fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=18, ha='right')
    ax2.set_ylim(-0.18, 0.72)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    for rect in rects3:
        h = rect.get_height()
        if h > 0.25:
            ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h - 0.035), xytext=(0, 0),
                         textcoords='offset points', ha='center', va='top', rotation=90,
                         fontsize=12, fontweight='bold', color='white')
        else:
            ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                         textcoords='offset points', ha='center', va='bottom', rotation=90,
                         fontsize=12, fontweight='bold', color='#102a45')
    for rect in rects4:
        h = rect.get_height()
        if h > 0.25:
            ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h - 0.035), xytext=(0, 0),
                         textcoords='offset points', ha='center', va='top', rotation=90,
                         fontsize=12, fontweight='bold', color='white')
        elif h >= 0:
            ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                         textcoords='offset points', ha='center', va='bottom', rotation=90,
                         fontsize=12, color='#a04000', fontweight='bold')
        else:
            ax2.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, -4),
                         textcoords='offset points', ha='center', va='top', rotation=90,
                         fontsize=12, color='#a04000', fontweight='bold')

    # Legenda Superior Unificada
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.035),
               ncol=2, frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc',
               prop={'family': 'sans-serif', 'size': 15, 'weight': 'bold'})

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    OUTPUT_INTERNAL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_INTERNAL, dpi=300, bbox_inches='tight')
    print(f'[OK] Gravado: {OUTPUT_INTERNAL}')

    if OUTPUT_THESIS.parent.exists():
        OUTPUT_THESIS.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_THESIS, dpi=300, bbox_inches='tight')
        print(f'[OK] Gravado: {OUTPUT_THESIS}')

    plt.close(fig)

if __name__ == '__main__':
    gerar_figura()
