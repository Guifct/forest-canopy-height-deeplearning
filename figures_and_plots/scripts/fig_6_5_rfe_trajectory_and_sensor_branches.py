"""
Figura 6.5 da Tese: Trajetória do RFE e Alocação dos Ramos de Sensores
Ficheiro de saída: rfe_pruning_and_sensor_branches.png

Painel Esquerdo: Trajetória do RMSE e R2 ao longo da eliminação recursiva de bandas (RFE),
                 evidenciando o ponto ótimo (sweet spot) em 42 bandas ativas (RMSE = 5.49m, R2 = 0.497).
Painel Direito:  Distribuição das 42 variáveis selecionadas entre os 3 ramos independentes
                 de sensores (Sentinel-1 SAR, Sentinel-2 Ótico e ALOS-2 SAR em banda L),
                 com destaque para a redução de 86% de dimensionalidade face às 302 bandas iniciais.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_fig_6_5(output_dirs=None):
    base_dir = Path(__file__).resolve().parent
    if output_dirs is None:
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir.parent / "organizado" / "RFE" / "RFE_results" / "rfe_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Resumo RFE não encontrado em: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Converter RMSE para metros se estiver em cm (> 50)
    df["rmse_m"] = df["mean_rmse"] / 100.0 if df["mean_rmse"].max() > 50 else df["mean_rmse"]

    # Configuração de estilo consistente com a tese
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 16,
        'axes.labelsize': 20,
        'xtick.labelsize': 17.5,
        'ytick.labelsize': 17.5,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.8))

    # ==========================================================
    # Painel Esquerdo: Trajetória de Poda RFE
    # ==========================================================
    color_rmse = '#b83b26'  # Vermelho escuro
    color_r2 = '#1f4e79'    # Azul escuro
    color_opt = '#27ae60'   # Verde ótimo

    # Eixo primário (RMSE)
    l1 = ax1.plot(df["n_bands"], df["rmse_m"], color=color_rmse, marker='o', linewidth=2.6, markersize=7, label='CV RMSE (m)')
    ax1.set_xlabel('Number of Active Predictor Bands', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Mean CV RMSE (m)', color=color_rmse, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color_rmse, labelsize=17.5)
    ax1.tick_params(axis='x', labelsize=17.5)
    ax1.set_xlim(310, -5)  # Invertido: 302 -> 0
    ax1.set_ylim(5.42, 5.86)
    ax1.set_yticks([5.45, 5.50, 5.55, 5.60, 5.65, 5.70, 5.75, 5.80, 5.85])
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Eixo secundário (R2)
    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(df["n_bands"], df["mean_r2"], color=color_r2, marker='s', linewidth=2.6, markersize=7, label=r'CV $R^2$')
    ax1_twin.set_ylabel(r'Coefficient of Determination ($R^2$)', color=color_r2, fontweight='bold', labelpad=10)
    ax1_twin.tick_params(axis='y', labelcolor=color_r2, labelsize=17.5)
    ax1_twin.set_ylim(0.435, 0.520)
    ax1_twin.set_yticks([0.44, 0.46, 0.48, 0.50, 0.52])
    ax1_twin.grid(False)

    # Ponto ótimo em 42 bandas
    opt_row = df[df["n_bands"] == 42].iloc[0]
    ax1.axvline(42, color=color_opt, linestyle='--', linewidth=2.4, alpha=0.9)
    opt_pt = ax1.scatter(42, opt_row["rmse_m"], color=color_opt, edgecolors='black', s=180, zorder=6,
                         label=f'Optimal (42 Bands, {opt_row["rmse_m"]:.2f}m)')

    lines = l1 + l2 + [opt_pt]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#555555', framealpha=1.0,
               fontsize=15.5, borderpad=0.6, labelspacing=0.4)

    # ==========================================================
    # Painel Direito: Alocação por Ramo de Sensores
    # ==========================================================
    sensor_names = [
        'Sentinel-1 SAR\n(C-Band)',
        'Sentinel-2 Optical\n(Multispectral)',
        'ALOS-2 SAR\n(L-Band)'
    ]
    counts = [19, 19, 4]
    pcts = [c / 42 * 100 for c in counts]
    colors = ['#1f77b4', '#2ca02c', '#8e44ad']

    bars = ax2.bar(sensor_names, counts, color=colors, edgecolor='black', linewidth=1.2, width=0.52)
    ax2.set_ylabel('Number of Retained Bands', fontweight='bold', labelpad=10)
    ax2.set_ylim(0, 26)
    ax2.set_yticks([0, 5, 10, 15, 20, 25])
    ax2.tick_params(axis='y', labelsize=17.5)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    for bar, c, p in zip(bars, counts, pcts):
        h = bar.get_height()
        ax2.annotate(f'{c} bands\n({p:.1f}%)', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 7), textcoords='offset points', ha='center', va='bottom',
                    fontweight='bold', fontsize=16)

    fig.tight_layout()

    filename = "rfe_pruning_and_sensor_branches.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_5()
