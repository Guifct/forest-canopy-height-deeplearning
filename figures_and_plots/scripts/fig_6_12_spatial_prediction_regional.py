"""
Figura 6.12 da Tese: Validação Espacial Regional das Predições de Altura (Serra da Lousã e Mafra)
Ficheiro de saída: spatial_prediction_regional_validation.png

Compara a dispersão entre alturas observadas GEDI RH98 e as alturas estimadas pelo modelo
Late Fusion (42 bandas RFE) em dois ecossistemas contrastantes de Portugal:
- Serra da Lousã (Relevo montanhoso acidentado, N = 5 376 pegadas)
- Mafra (Floresta costeira atlântica, N = 414 pegadas)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def generate_fig_6_12(output_dirs=None):
    base_dir = Path(__file__).resolve().parent
    if output_dirs is None:
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir.parent / "organizado" / "RFE" / "Final_Model_42_Bands" / "final_42_bands_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dados das 42 bandas não encontrados em: {csv_path}")

    df = pd.read_csv(csv_path)
    # Converter centímetros para metros se necessário
    df["yt_m"] = df["y_true"] / 100.0 if df["y_true"].max() > 80 else df["y_true"]
    df["yp_m"] = df["y_pred"] / 100.0 if df["y_pred"].max() > 80 else df["y_pred"]

    lousa = df[df["region"].isin(["SerraLousa", "Serra da Lousã", "Serra_da_Lousa"])].copy()
    mafra = df[df["region"] == "Mafra"].copy()

    # Configuração de estilo consistente com a tese
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Paleta YlGn calibrada para garantir visibilidade das alturas baixas (<8m)
    ylgn_base = plt.colormaps['YlGn']
    colors_ylgn = ylgn_base(np.linspace(0.20, 0.95, 256))
    custom_ylgn = LinearSegmentedColormap.from_list('enhanced_ylgn', colors_ylgn)

    # Hierarquia tipográfica calibrada para inserção a \textwidth
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13.5,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 6.2))

    # ==========================================================
    # Painel Esquerdo: Serra da Lousã (N = 5 376)
    # ==========================================================
    sc1 = ax1.scatter(lousa["yt_m"], lousa["yp_m"], c=lousa["yp_m"], cmap=custom_ylgn,
                      alpha=0.55, s=18, vmin=0, vmax=35, edgecolors='none', zorder=3)
    ax1.plot([0, 45], [0, 45], 'k--', linewidth=2.0, alpha=0.9, label='1:1 Ideal Line', zorder=4)

    ax1.set_xlabel(r'Observed GEDI $\mathrm{RH}_{98}$ Height (m)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Predicted Canopy Height (m)', fontweight='bold', labelpad=10)
    ax1.set_title(r'Serra da Lousã ($N = 5\,376$ Footprints)', fontweight='bold', pad=12)
    ax1.set_xlim(0, 45)
    ax1.set_ylim(0, 45)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

    cb1 = fig.colorbar(sc1, ax=ax1, pad=0.02)
    cb1.set_label('Height (m)', fontweight='bold', fontsize=14, labelpad=8)
    cb1.ax.tick_params(labelsize=13)

    # ==========================================================
    # Painel Direito: Mafra (N = 414)
    # ==========================================================
    sc2 = ax2.scatter(mafra["yt_m"], mafra["yp_m"], c=mafra["yp_m"], cmap=custom_ylgn,
                      alpha=0.65, s=26, vmin=0, vmax=35, edgecolors='none', zorder=3)
    ax2.plot([0, 45], [0, 45], 'k--', linewidth=2.0, alpha=0.9, label='1:1 Ideal Line', zorder=4)

    ax2.set_xlabel(r'Observed GEDI $\mathrm{RH}_{98}$ Height (m)', fontweight='bold', labelpad=10)
    ax2.set_ylabel('Predicted Canopy Height (m)', fontweight='bold', labelpad=10)
    ax2.set_title(r'Mafra ($N = 414$ Footprints)', fontweight='bold', pad=12)
    ax2.set_xlim(0, 45)
    ax2.set_ylim(0, 45)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

    cb2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cb2.set_label('Height (m)', fontweight='bold', fontsize=14, labelpad=8)
    cb2.ax.tick_params(labelsize=13)

    fig.tight_layout()

    filename = "spatial_prediction_regional_validation.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_12()
