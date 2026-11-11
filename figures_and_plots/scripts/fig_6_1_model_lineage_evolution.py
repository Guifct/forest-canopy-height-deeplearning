"""
Figura 6.1 da Tese: Evolução Cronológica da Linhagem dos Modelos (Table 6.1)
Ficheiro de saída: model_lineage_evolution_bar_chart.png

Este script plota a progressão dos 7 patamares metodológicos avaliados na tese:
1. Dummy Regressor (Baseline de média constante)
2. Minimalist CNN (Early Fusion superficial)
3. SpatialCNN (Early Fusion completo com 302 bandas)
4. LateFusionSpatialCNN (Fusão tardia modular com 42 bandas RFE)
5. + Center-Aware (Pooling com priorização central)
6. + SE Attention (Atenção Squeeze-and-Excitation por sensor)
7. Multi-Seed Ensemble (Ensemble estocástico multi-seed)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def generate_fig_6_1(output_dirs=None):
    if output_dirs is None:
        base_dir = Path(__file__).resolve().parent
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Configuração de estilo consistente com a tese
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.dpi': 300
    })

    tiers = [
        '1. Dummy\nRegressor',
        '2. Minimalist\nCNN',
        '3. SpatialCNN\n(302 Bands)',
        '4. LateFusion\n(42 Bands)',
        '5. + Center\nAware',
        '6. + SE\nAttention',
        '7. Multi-Seed\nEnsemble'
    ]
    pct_rmse_vals = [49.5, 47.5, 35.8, 35.1, 34.7, 34.6, 34.1]
    r2_vals       = [0.000, 0.067, 0.475, 0.500, 0.511, 0.514, 0.526]
    x_idx = np.arange(len(tiers))

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    color_rmse = '#204a87'  # Azul escuro
    color_r2   = '#2ca02c'  # Verde

    # Curva %RMSE
    line1 = ax1.plot(x_idx, pct_rmse_vals, color=color_rmse, marker='o', linewidth=2.8, markersize=8.5, label=r'Validation %RMSE (%)', zorder=3)
    ax1.set_ylabel(r'Validation %RMSE (%)', color=color_rmse, fontweight='bold', labelpad=8)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.set_ylim(28.0, 54.0)
    ax1.grid(True, linestyle='--', alpha=0.5)

    for i, txt in enumerate(pct_rmse_vals):
        y_off = 10
        ax1.annotate(f'{txt:.1f}%', (x_idx[i], pct_rmse_vals[i]), textcoords="offset points", xytext=(0, y_off), ha='center', fontweight='bold', fontsize=12, color=color_rmse)

    # Curva R2
    ax2 = ax1.twinx()
    line2 = ax2.plot(x_idx, r2_vals, color=color_r2, marker='D', linewidth=2.8, markersize=8.5, label=r'Coefficient of Determination ($R^2$)', zorder=3)
    ax2.set_ylabel(r'Coefficient of Determination ($R^2$)', color=color_r2, fontweight='bold', labelpad=8)
    ax2.tick_params(axis='y', labelcolor=color_r2)
    ax2.set_ylim(-0.04, 0.74)
    ax2.grid(False)

    for i, txt in enumerate(r2_vals):
        y_off = 10 if i in [0, 1] else 12
        ax2.annotate(f'{txt:.3f}', (x_idx[i], r2_vals[i]), textcoords="offset points", xytext=(0, y_off), ha='center', fontweight='bold', fontsize=12, color=color_r2)

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(tiers, fontweight='medium')
    ax1.set_xlabel('Model Lineage Progression', fontweight='bold', labelpad=12)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95)

    fig.tight_layout()

    filename = "model_lineage_evolution_bar_chart.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_1()
