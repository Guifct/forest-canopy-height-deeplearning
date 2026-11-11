"""
Figura 6.11 da Tese:
Curvas de Aprendizagem Comparativas: Augmentation OFF vs. Augmentation ON
Legenda da Tese:
"Comparative 5-fold cross-validation learning trajectories across training epochs,
illustrating dual-axis validation RMSE (blue solid, left axis) and validation R2
(green dashed, right axis):
(Left) Augmentation OFF, exhibiting initial error minimization at epoch 10 followed
by persistent validation loss divergence and overfitting up to epoch 32;
(Right) Augmentation ON, demonstrating robust, monotonic convergence and sustained
variance explanation through all 40 epochs."

Ficheiros gerados:
1. learning_curves_aug_off_vs_on_left.png (Painel esquerdo: Augmentation OFF)
2. learning_curves_aug_off_vs_on_right.png (Painel direito: Augmentation ON)
3. learning_curves_aug_off_vs_on.png (Painel combinado 1x2 lado a lado)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

OUTPUT_INTERNAL = BASE_DIR / "outputs"
OUTPUT_THESIS = REPO_ROOT.parent / "Escrita_tese_Rascunho" / "5-Figures" / "Results"

def gerar_figuras():
    for d in [OUTPUT_INTERNAL, OUTPUT_THESIS]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Dados Experimentais Reais de Treino e Validação (5-Fold CV)
    # Citados explicitamente na Secção 6.3.2 da Tese
    # -------------------------------------------------------------
    epochs_off = np.arange(1, 33)
    rmse_off = np.array([
        7.693, 6.792, 6.117, 5.915, 5.927, 5.796, 5.743, 5.736, 5.719, 5.670,
        5.737, 5.758, 5.714, 5.691, 5.755, 5.763, 5.732, 5.747, 5.750, 5.792,
        5.840, 5.833, 5.879, 5.869, 5.898, 5.949, 5.976, 6.014, 6.009, 5.991,
        5.989, 6.030
    ])
    r2_off = np.array([
       -0.015, 0.210, 0.374, 0.416, 0.413, 0.440, 0.451, 0.452, 0.455, 0.465,
        0.452, 0.447, 0.456, 0.461, 0.447, 0.446, 0.452, 0.449, 0.449, 0.440,
        0.431, 0.432, 0.423, 0.425, 0.420, 0.416, 0.412, 0.405, 0.410, 0.432,
        0.435, 0.428
    ])

    epochs_on = np.arange(1, 41)
    rmse_on = np.array([
        7.753, 6.818, 6.172, 6.024, 5.961, 5.875, 5.810, 5.815, 5.813, 5.783,
        5.768, 5.764, 5.771, 5.716, 5.705, 5.703, 5.689, 5.677, 5.671, 5.687,
        5.684, 5.710, 5.708, 5.664, 5.655, 5.675, 5.680, 5.662, 5.668, 5.667,
        5.650, 5.654, 5.655, 5.654, 5.634, 5.622, 5.669, 5.633, 5.662, 5.656
    ])
    r2_on = np.array([
       -0.024, 0.209, 0.361, 0.393, 0.406, 0.423, 0.436, 0.435, 0.435, 0.442,
        0.445, 0.446, 0.444, 0.455, 0.457, 0.458, 0.460, 0.463, 0.464, 0.461,
        0.462, 0.456, 0.457, 0.465, 0.467, 0.463, 0.462, 0.466, 0.465, 0.465,
        0.468, 0.467, 0.467, 0.469, 0.476, 0.480, 0.470, 0.477, 0.472, 0.473
    ])

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.labelsize': 16,
        'xtick.labelsize': 13.5,
        'ytick.labelsize': 13.5,
        'figure.dpi': 300
    })

    color_rmse = '#1f4e79'  # Azul marinho profundo
    color_r2   = '#27ae60'  # Verde limpo

    # ==========================================================
    # 1. Painel Esquerdo: Augmentation OFF (Salvo individualmente)
    # ==========================================================
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.4), dpi=300)
    l1 = ax1.plot(epochs_off, rmse_off, color=color_rmse, linewidth=2.6,
                  marker='o', markersize=5.5, label='Validation RMSE (m)', zorder=4)
    ax1.set_xlabel('Training Epochs (Augmentation OFF)', fontsize=15.5, fontweight='bold', labelpad=9)
    ax1.set_ylabel('Validation RMSE (m)', color=color_rmse, fontsize=16, fontweight='bold', labelpad=9)
    ax1.tick_params(axis='y', labelcolor=color_rmse, labelsize=13.5)
    ax1.tick_params(axis='x', labelsize=13.5)
    ax1.set_ylim(5.3, 8.2)
    ax1.set_xlim(0, 42)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_t = ax1.twinx()
    l2 = ax1_t.plot(epochs_off, r2_off, color=color_r2, linewidth=2.6,
                    linestyle='--', marker='s', markersize=5.5, label=r'Validation $R^2$', zorder=4)
    ax1_t.set_ylabel(r'Validation $R^2$', color=color_r2, fontsize=16, fontweight='bold', labelpad=9)
    ax1_t.tick_params(axis='y', labelcolor=color_r2, labelsize=13.5)
    ax1_t.set_ylim(-0.08, 0.55)
    ax1_t.grid(False)

    lines_p1 = l1 + l2
    ax1.legend(lines_p1, [l.get_label() for l in lines_p1], loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor='white',
               framealpha=1.0, edgecolor='#7f8c8d', fontsize=13, columnspacing=1.8)
    fig1.tight_layout()

    # ==========================================================
    # 2. Painel Direito: Augmentation ON (Salvo individualmente)
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.4), dpi=300)
    l3 = ax2.plot(epochs_on, rmse_on, color=color_rmse, linewidth=2.6,
                  marker='o', markersize=5.5, label='Validation RMSE (m)', zorder=4)
    ax2.set_xlabel('Training Epochs (Augmentation ON)', fontsize=15.5, fontweight='bold', labelpad=9)
    ax2.set_ylabel('Validation RMSE (m)', color=color_rmse, fontsize=16, fontweight='bold', labelpad=9)
    ax2.tick_params(axis='y', labelcolor=color_rmse, labelsize=13.5)
    ax2.tick_params(axis='x', labelsize=13.5)
    ax2.set_ylim(5.3, 8.2)
    ax2.set_xlim(0, 42)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2_t = ax2.twinx()
    l4 = ax2_t.plot(epochs_on, r2_on, color=color_r2, linewidth=2.6,
                    linestyle='--', marker='s', markersize=5.5, label=r'Validation $R^2$', zorder=4)
    ax2_t.set_ylabel(r'Validation $R^2$', color=color_r2, fontsize=16, fontweight='bold', labelpad=9)
    ax2_t.tick_params(axis='y', labelcolor=color_r2, labelsize=13.5)
    ax2_t.set_ylim(-0.08, 0.55)
    ax2_t.grid(False)

    lines_p2 = l3 + l4
    ax2.legend(lines_p2, [l.get_label() for l in lines_p2], loc='lower center',
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor='white',
               framealpha=1.0, edgecolor='#7f8c8d', fontsize=13, columnspacing=1.8)
    fig2.tight_layout()

    # ==========================================================
    # 3. Painel Combinado 1x2 (Lado a Lado)
    # ==========================================================
    fig_comb, (c_ax1, c_ax2) = plt.subplots(1, 2, figsize=(14.8, 5.6), dpi=300)

    # Sub-painel 1
    cl1 = c_ax1.plot(epochs_off, rmse_off, color=color_rmse, linewidth=2.6, marker='o', markersize=5.5, label='Validation RMSE (m)', zorder=4)
    c_ax1.set_xlabel('Training Epochs (Augmentation OFF)', fontsize=16, fontweight='bold', labelpad=9)
    c_ax1.set_ylabel('Validation RMSE (m)', color=color_rmse, fontsize=16, fontweight='bold', labelpad=9)
    c_ax1.tick_params(axis='y', labelcolor=color_rmse, labelsize=13.5)
    c_ax1.tick_params(axis='x', labelsize=13.5)
    c_ax1.set_ylim(5.3, 8.2); c_ax1.set_xlim(0, 42)
    c_ax1.grid(True, linestyle='--', alpha=0.5)

    c_ax1_t = c_ax1.twinx()
    cl2 = c_ax1_t.plot(epochs_off, r2_off, color=color_r2, linewidth=2.6, linestyle='--', marker='s', markersize=5.5, label=r'Validation $R^2$', zorder=4)
    c_ax1_t.set_ylabel(r'Validation $R^2$', color=color_r2, fontsize=16, fontweight='bold', labelpad=9)
    c_ax1_t.tick_params(axis='y', labelcolor=color_r2, labelsize=13.5)
    c_ax1_t.set_ylim(-0.08, 0.55); c_ax1_t.grid(False)

    # Sub-painel 2
    cl3 = c_ax2.plot(epochs_on, rmse_on, color=color_rmse, linewidth=2.6, marker='o', markersize=5.5, label='Validation RMSE (m)', zorder=4)
    c_ax2.set_xlabel('Training Epochs (Augmentation ON)', fontsize=16, fontweight='bold', labelpad=9)
    c_ax2.set_ylabel('Validation RMSE (m)', color=color_rmse, fontsize=16, fontweight='bold', labelpad=9)
    c_ax2.tick_params(axis='y', labelcolor=color_rmse, labelsize=13.5)
    c_ax2.tick_params(axis='x', labelsize=13.5)
    c_ax2.set_ylim(5.3, 8.2); c_ax2.set_xlim(0, 42)
    c_ax2.grid(True, linestyle='--', alpha=0.5)

    c_ax2_t = c_ax2.twinx()
    cl4 = c_ax2_t.plot(epochs_on, r2_on, color=color_r2, linewidth=2.6, linestyle='--', marker='s', markersize=5.5, label=r'Validation $R^2$', zorder=4)
    c_ax2_t.set_ylabel(r'Validation $R^2$', color=color_r2, fontsize=16, fontweight='bold', labelpad=9)
    c_ax2_t.tick_params(axis='y', labelcolor=color_r2, labelsize=13.5)
    c_ax2_t.set_ylim(-0.08, 0.55); c_ax2_t.grid(False)

    lines_comb = cl1 + cl2
    labels_comb = [l.get_label() for l in lines_comb]
    fig_comb.legend(lines_comb, labels_comb, loc='lower center', bbox_to_anchor=(0.5, 0.98),
                    ncol=2, frameon=True, facecolor='white', framealpha=1.0, edgecolor='#7f8c8d',
                    fontsize=14.5, columnspacing=2.5)

    fig_comb.tight_layout()

    # Gravação em todas as pastas de saída
    for out_dir in [OUTPUT_INTERNAL, OUTPUT_THESIS]:
        f_left = out_dir / "learning_curves_aug_off_vs_on_left.png"
        f_right = out_dir / "learning_curves_aug_off_vs_on_right.png"
        f_comb = out_dir / "learning_curves_aug_off_vs_on.png"

        fig1.savefig(f_left, dpi=300, bbox_inches='tight')
        fig2.savefig(f_right, dpi=300, bbox_inches='tight')
        fig_comb.savefig(f_comb, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {f_left}")
        print(f"[OK] Gravado: {f_right}")
        print(f"[OK] Gravado: {f_comb}")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig_comb)

if __name__ == '__main__':
    gerar_figuras()
