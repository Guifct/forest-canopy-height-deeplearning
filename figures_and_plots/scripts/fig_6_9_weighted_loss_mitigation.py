"""
Figura 6.9 da Tese: Mitigação do Viés e Redução do RMSE com High-Height Weighted Loss
Ficheiro de saída: weighted_loss_rmse_reduction.png

Painel Esquerdo: Mitigação do viés sistemático (Mean Bias em metros) por estrato de altura
                 comparando Standard MSE Loss vs High-Height Weighted Loss (w = 2.0).
Painel Direito:  Redução do erro quadrático médio (RMSE em metros) nos estratos de vegetação,
                 demonstrando o ganho expressivo nas copas maduras e altas (>25m: 7.42m -> 4.85m).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def generate_fig_6_9(output_dirs=None):
    base_dir = Path(__file__).resolve().parent
    if output_dirs is None:
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Configuração de estilo consistente com a tese (alta legibilidade)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 16,
        'axes.labelsize': 20,
        'xtick.labelsize': 17.5,
        'ytick.labelsize': 17.5,
        'figure.dpi': 300
    })

    tiers_labels = ['< 5m\n(Short)', '5–15m\n(Interm.)', '15–25m\n(Mature)', '> 25m\n(Tall)']
    std_bias = [+2.18, +0.42, -0.88, -5.92]
    wgt_bias = [+0.85, +0.11, -0.24, -1.35]

    std_rmse = [3.12, 2.78, 3.84, 7.42]
    wgt_rmse = [2.64, 2.55, 3.41, 4.85]

    x = np.arange(len(tiers_labels))
    bar_w = 0.32
    gap = 0.035

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.0))

    c_std = '#d9534f'  # Vermelho Standard MSE
    c_wgt = '#27ae60'  # Verde Weighted Loss

    # ==========================================================
    # Painel 1: Mitigação do Viés (Mean Bias)
    # ==========================================================
    r1 = ax1.bar(x - bar_w/2 - gap/2, std_bias, bar_w, label='Standard MSE Loss', color=c_std, edgecolor='black', linewidth=1.1)
    r2 = ax1.bar(x + bar_w/2 + gap/2, wgt_bias, bar_w, label='High-Height Weighted Loss (Proposed)', color=c_wgt, edgecolor='black', linewidth=1.1)

    ax1.axhline(0, color='black', linestyle='-', linewidth=1.1, alpha=0.8)
    ax1.set_ylabel('Prediction Bias (m)', fontweight='bold', labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers_labels, fontsize=16.5)
    ax1.set_ylim(-7.4, 3.8)
    ax1.tick_params(axis='both', which='major', labelsize=17.5)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    for r in r1:
        h = r.get_height()
        va = 'bottom' if h >= 0 else 'top'
        y_off = 0.22 if h >= 0 else -0.32
        ax1.annotate(f'{h:+.2f}m', xy=(r.get_x() + r.get_width()/2, h + y_off),
                     ha='center', va=va, fontsize=15, fontweight='bold', color='#a93226')

    for r in r2:
        h = r.get_height()
        va = 'bottom' if h >= 0 else 'top'
        y_off = 0.22 if h >= 0 else -0.32
        ax1.annotate(f'{h:+.2f}m', xy=(r.get_x() + r.get_width()/2, h + y_off),
                     ha='center', va=va, fontsize=15, fontweight='bold', color='#196f3d')

    # ==========================================================
    # Painel 2: Redução do RMSE
    # ==========================================================
    r3 = ax2.bar(x - bar_w/2 - gap/2, std_rmse, bar_w, label='Standard MSE Loss', color=c_std, edgecolor='black', linewidth=1.1)
    r4 = ax2.bar(x + bar_w/2 + gap/2, wgt_rmse, bar_w, label='High-Height Weighted Loss (Proposed)', color=c_wgt, edgecolor='black', linewidth=1.1)

    ax2.set_ylabel('RMSE (m)', fontweight='bold', labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers_labels, fontsize=16.5)
    ax2.set_ylim(0, 9.4)
    ax2.tick_params(axis='both', which='major', labelsize=17.5)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    for r in r3:
        h = r.get_height()
        ax2.annotate(f'{h:.2f}m', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                     fontsize=15, fontweight='bold', color='#a93226')

    for r in r4:
        h = r.get_height()
        ax2.annotate(f'{h:.2f}m', xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                     fontsize=15, fontweight='bold', color='#196f3d')

    # Legenda unificada no topo
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.98),
               ncol=2, frameon=True, facecolor='white', edgecolor='#555555',
               framealpha=1.0, fontsize=16.5, borderpad=0.6, columnspacing=2.5)

    fig.tight_layout()

    filename = "weighted_loss_rmse_reduction.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_9()
