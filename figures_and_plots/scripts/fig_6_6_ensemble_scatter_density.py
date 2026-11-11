"""
Figura 6.6 da Tese: Dispersão e Densidade 2D do Multi-Seed Ensemble Proposto
Ficheiro de saída: ensemble_final_ca_scatter.png

Gráfico de dispersão e mapa de densidade 2D de amostras (colormap magma com barra superior)
comparando a altura observada GEDI RH98 vs a altura prevista pelo Ensemble Final (Late Fusion + Center-Aware + SE),
destacando a linha ideal 1:1 e a reta de regressão linear (y = 0.53x + 7.13 / 7.17).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_fig_6_6(output_dirs=None):
    base_dir = Path(__file__).resolve().parent
    if output_dirs is None:
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir.parent / "organizado" / "cross_validation" / "random seed" / "outputs_fch" / "spacial_cnn" / "late_fusion_cnn_11x11_SE_attention_ensemble" / "predicoes_ensemble.csv"
    if not csv_path.exists():
        # Tentar caminho alternativo no center_aware_ensemble
        csv_path = base_dir.parent / "organizado" / "cross_validation" / "random seed" / "outputs_fch" / "spacial_cnn" / "late_fusion_cnn_11x11_center_aware_ensemble" / "predicoes_ensemble.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dados do ensemble não encontrados em: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Selecionar o modelo standard ensemble
    target_variants = ["se_standard_ensemble3", "ca_standard_ensemble3"]
    selected_variant = next((v for v in target_variants if v in df["model_variant"].values), df["model_variant"].iloc[0])
    sub = df[df["model_variant"] == selected_variant].copy()

    y_true = sub["y_true_m"].to_numpy()
    y_pred = sub["y_pred_m"].to_numpy()

    # Reta de regressão
    slope, intercept = np.polyfit(y_true, y_pred, 1)

    # Configurações de estilo para publicação
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

    fig = plt.figure(figsize=(7.8, 7.8))
    ax = fig.add_axes([0.12, 0.10, 0.82, 0.76])

    # Histograma 2D com magma colormap
    bins = 72
    counts, xedges, yedges = np.histogram2d(y_true, y_pred, bins=bins, range=[[0, 45], [0, 45]])
    counts = np.ma.masked_where(counts == 0, counts)

    mesh = ax.pcolormesh(xedges, yedges, counts.T, cmap='magma', shading='flat', vmin=1, vmax=25)

    # Linha ideal 1:1 (tracejada verde de alta visibilidade)
    line_ideal, = ax.plot([0, 45], [0, 45], linestyle='--', color='#00c853', linewidth=3.8, label=r'1:1 Ideal Line ($y = x$)', zorder=5)

    # Reta de tendência ajustada (linha contínua vermelha)
    x_vals = np.array([0, 45])
    y_vals = slope * x_vals + intercept
    line_fit, = ax.plot(x_vals, y_vals, linestyle='-', color='#d90429', linewidth=3.5, label=f'Regression Trendline ($y = {slope:.2f}x + {intercept:.2f}$)', zorder=6)

    ax.set_xlabel(r'Observed GEDI $\mathrm{RH}_{98}$ Canopy Height (m)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Predicted Canopy Height (m)', fontweight='bold', labelpad=10)
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 45)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.45)

    ax.legend(handles=[line_ideal, line_fit], loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#7f8c8d')

    # Barra de cores posicionada no topo
    cax = fig.add_axes([0.12, 0.90, 0.82, 0.032])
    cbar = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cbar.set_label('Sample Density (Number of Observations)', fontweight='bold', labelpad=8)
    cbar.set_ticks([5, 10, 15, 20, 25])

    filename = "ensemble_final_ca_scatter.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_6()
