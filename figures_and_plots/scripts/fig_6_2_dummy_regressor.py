"""
Figura 6.2 da Tese: Visualização do Baseline Dummy Regressor
Ficheiro de saída: dummy_regressor_visualisation.png

Painel Esquerdo: Gráfico de dispersão real vs previsto demonstrando o colapso das previsões
                 numa linha horizontal constante correspondente à média global.
Painel Direito:  Histograma da densidade de probabilidade mostrando a distribuição real GEDI RH98
                 em contraste com o pico impulsional da média constante do Dummy Regressor.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_fig_6_2(output_dirs=None):
    base_dir = Path(__file__).resolve().parent
    if output_dirs is None:
        output_dirs = [
            base_dir / "outputs",
            Path(r"c:\Users\gui\Documents\Tese\Escrita_tese_Rascunho\5-Figures\Results")
        ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir.parent / "organizado" / "cross_validation" / "outputs_fch" / "spacial_cnn" / "cv_global_misturado_estratificado" / "predicoes_cv_global_misturado_ablation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dados não encontrados em: {csv_path}")

    df = pd.read_csv(csv_path)
    df_aug = df[df["augmentation"] == "aug_on"].copy()
    y_true = df_aug["y_true_m"].to_numpy()
    N = len(y_true)
    mean_height = float(np.mean(y_true))

    y_dummy = np.full(N, mean_height)

    # Configuração de estilo com tipografia extra-grande consistente com a tese
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 16,
        'axes.labelsize': 20,
        'xtick.labelsize': 17.5,
        'ytick.labelsize': 17.5,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6))

    # Painel 1: Scatter plot
    np.random.seed(42)
    sample_idx = np.random.choice(N, min(1000, N), replace=False)
    ax1.scatter(y_true[sample_idx], y_dummy[sample_idx], alpha=0.85, color='#d4ac0d', edgecolors='#9a7d0a', linewidths=0.7, s=36, label='Dummy Predictions', zorder=4)
    ax1.plot([0, 50], [0, 50], color='black', linestyle='--', linewidth=1.8, alpha=0.85, label='1:1 Ideal Line', zorder=2)
    ax1.axhline(mean_height, color='#c0392b', linestyle='-', linewidth=3.2, label=f'Mean ({mean_height:.2f} m)', zorder=3)
    ax1.set_xlabel(r'Observed GEDI $\mathrm{RH}_{98}$ Height (m)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Predicted Canopy Height (m)', fontweight='bold', labelpad=10)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)
    ax1.tick_params(axis='both', which='major', labelsize=17.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#555555', framealpha=1.0, fontsize=16.5, borderpad=0.7, labelspacing=0.5)

    # Painel 2: Histograma de densidade
    bins = np.linspace(0, 50, 46)
    ax2.hist(y_true, bins=bins, density=True, alpha=0.65, color='#4a7ab7', label='Target GEDI', edgecolor='#2c5282', linewidth=0.8)
    ax2.axvline(mean_height, color='#c0392b', linewidth=3.2, label=f'Mean Peak ({mean_height:.2f} m)')
    ax2.set_xlabel('Canopy Height (m)', fontweight='bold', labelpad=10)
    ax2.set_ylabel('Probability Density', fontweight='bold', labelpad=10)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 0.053)
    ax2.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax2.set_yticklabels(['0.00', '0.01', '0.02', '0.03', '0.04', '0.05'])
    ax2.tick_params(axis='both', which='major', labelsize=17.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#555555', framealpha=1.0, fontsize=16.5, borderpad=0.7, labelspacing=0.5)

    fig.tight_layout()

    filename = "dummy_regressor_visualisation.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_2()
