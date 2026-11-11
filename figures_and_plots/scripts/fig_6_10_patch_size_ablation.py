"""
Figura 6.10 da Tese: Estudo de Ablação da Dimensão da Janela Espacial (Patch Size)
Ficheiro de saída: patch_size_ablation_comparison.png

Análise da sensibilidade à dimensão da janela espacial de entrada (7x7, 11x11 e 15x15
na grelha raster de 25m), evidenciando que a configuração intermediária 11x11 (275m x 275m)
atinge o equilíbrio ótimo entre campo recetivo contextual e fidelidade aos contornos florestais
(RMSE = 5.56m, R2 = 0.485).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def generate_fig_6_10(output_dirs=None):
    base_dir = Path(__file__).resolve().parent
    if output_dirs is None:
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Configuração de estilo consistente com a tese (tipografia refinada/discreta)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11.5,
        'axes.titlesize': 12.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })

    patch_sizes = ['7×7 (175m)', '11×11 (275m)\n[Optimal]', '15×15 (375m)']
    patch_rmse = [5.60, 5.56, 5.57]
    patch_r2   = [0.477, 0.485, 0.484]

    color_rmse = '#1f4e79'  # Azul escuro
    color_r2 = '#27ae60'    # Verde

    fig, ax1 = plt.subplots(figsize=(7.8, 4.8))

    # Eixo RMSE (esquerda)
    line1 = ax1.plot(patch_sizes, patch_rmse, marker='o', linewidth=2.5, markersize=7.5, color=color_rmse, label='Validation RMSE (m)', zorder=4)
    ax1.set_xlabel('Input Spatial Window Size', fontweight='bold', labelpad=9)
    ax1.set_ylabel('Validation RMSE (m)', color=color_rmse, fontweight='bold', labelpad=8)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.set_ylim(5.45, 5.70)
    ax1.set_yticks([5.50, 5.55, 5.60, 5.65, 5.70])
    ax1.grid(True, linestyle='--', alpha=0.5)

    for i, txt in enumerate(patch_rmse):
        y_off = 10 if i == 0 else -16
        ax1.annotate(f'{txt:.2f} m', (patch_sizes[i], patch_rmse[i]), textcoords='offset points',
                     xytext=(0, y_off), ha='center', fontweight='bold', fontsize=9.5, color=color_rmse)

    # Eixo R2 (direita)
    ax2 = ax1.twinx()
    line2 = ax2.plot(patch_sizes, patch_r2, marker='s', linewidth=2.5, linestyle='--', markersize=7.5, color=color_r2, label=r'Validation $R^2$', zorder=4)
    ax2.set_ylabel(r'Validation $R^2$', color=color_r2, fontweight='bold', labelpad=8)
    ax2.tick_params(axis='y', labelcolor=color_r2)
    ax2.set_ylim(0.470, 0.495)
    ax2.set_yticks([0.475, 0.480, 0.485, 0.490, 0.495])
    ax2.grid(False)

    for i, txt in enumerate(patch_r2):
        y_off = 10 if (i == 1 or i == 2) else -18
        ax2.annotate(f'{txt:.3f}', (patch_sizes[i], patch_r2[i]), textcoords='offset points',
                     xytext=(0, y_off), ha='center', fontweight='bold', fontsize=9.5, color=color_r2)

    # Faixa sombreada de destaque para a dimensão ótima (11x11)
    ax1.axvspan(0.7, 1.3, color='#27ae60', alpha=0.12, label='Optimal Receptive Field (11×11)', zorder=1)

    fig.tight_layout()

    filename = "patch_size_ablation_comparison.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_10()
