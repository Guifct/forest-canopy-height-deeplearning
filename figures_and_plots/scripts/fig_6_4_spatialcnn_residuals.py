"""
Figura 6.4 da Tese: Diagnóstico Empírico de Erros e Resíduos para o SpatialCNN
Ficheiro de saída: spatialcnn_error_residuals.png

Painel Esquerdo: Distribuição da densidade de probabilidade dos resíduos (e = y_pred - y_true)
                 com ajuste gaussiano teórico (mu = -0.27m, sigma = 5.57m).
Painel Direito:  Dispersão dos resíduos em função da altura observada GEDI RH98 (N = 8 864),
                 evidenciando a sobrestimação em vegetação baixa, a subestimação em copas altas
                 e as faixas de variância por classes (+- 1 sigma).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_fig_6_4(output_dirs=None):
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
        raise FileNotFoundError(f"Ficheiro de dados não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    df_aug = df[df["augmentation"] == "aug_on"].copy()
    y_true = df_aug["y_true_m"].to_numpy()
    y_pred = df_aug["y_pred_m"].to_numpy()
    residuals = df_aug["residual_m"].to_numpy()
    N = len(residuals)

    mu, sigma = float(np.mean(residuals)), float(np.std(residuals))

    # Configuração de estilo consistente com 6.2 e 6.3 para legibilidade perfeita na tese
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

    # ==========================================================
    # Painel Esquerdo: Histograma e Ajuste Gaussiano
    # ==========================================================
    bins_res = np.linspace(-25, 25, 51)
    ax1.hist(residuals, bins=bins_res, density=True, alpha=0.60, color='#3498db', edgecolor='#2980b9', label='Residuals')
    
    xs = np.linspace(-25, 25, 300)
    ax1.plot(xs, norm.pdf(xs, mu, sigma), color='#c0392b', linewidth=2.8, label=f'Gaussian ($\mu = {mu:.2f}, \sigma = {sigma:.2f}\\mathrm{{m}}$)')
    ax1.plot([0, 0], [0, 0.072], color='black', linestyle='--', linewidth=1.6, label='Zero Error ($e = 0$)')
    ax1.plot([mu, mu], [0, 0.072], color='#c0392b', linestyle=':', linewidth=2.2, label=f'Mean Bias ({mu:.2f}m)')

    ax1.set_xlabel(r'Residual $e = y_{\mathrm{pred}} - y_{\mathrm{true}}$ (m)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Probability Density', fontweight='bold', labelpad=10)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(0, 0.130)
    ax1.set_yticks([0.00, 0.03, 0.06, 0.09, 0.12])
    ax1.set_yticklabels(['0.00', '0.03', '0.06', '0.09', '0.12'])
    ax1.tick_params(axis='both', which='major', labelsize=17.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#555555', framealpha=1.0, 
               fontsize=15.5, borderpad=0.6, labelspacing=0.35)

    # ==========================================================
    # Painel Direito: Dispersão dos Resíduos e Médias por Classes
    # ==========================================================
    ax2.scatter(y_true, residuals, alpha=0.18, color='#34495e', s=16, edgecolors='none', label=f'Residuals ($N = {N:,}$)')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.6, label='Zero Error Baseline')

    # Cálculo dos bins de altura e desvio padrão
    bins_h = np.linspace(0, 45, 10)
    centers = 0.5 * (bins_h[:-1] + bins_h[1:])
    binned_means, binned_stds = [], []

    for i in range(len(bins_h) - 1):
        mask = (y_true >= bins_h[i]) & (y_true < bins_h[i + 1])
        binned_means.append(np.mean(residuals[mask]))
        binned_stds.append(np.std(residuals[mask]))

    binned_means = np.array(binned_means)
    binned_stds = np.array(binned_stds)

    ax2.plot(centers, binned_means, color='#e74c3c', marker='o', linewidth=2.8, markersize=7.5, label=r'Binned Mean $\mu(h)$')
    ax2.fill_between(centers, binned_means - binned_stds, binned_means + binned_stds, color='#e74c3c', alpha=0.25, label=r'Variance ($\pm 1\sigma$)')

    ax2.set_xlabel(r'Observed GEDI $\mathrm{RH}_{98}$ Height (m)', fontweight='bold', labelpad=10)
    ax2.set_ylabel(r'Residual $e = y_{\mathrm{pred}} - y_{\mathrm{true}}$ (m)', fontweight='bold', labelpad=10)
    ax2.set_xlim(0, 45)
    ax2.set_ylim(-30, 30)
    ax2.tick_params(axis='both', which='major', labelsize=17.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#555555', framealpha=1.0, 
               fontsize=15.5, borderpad=0.6, labelspacing=0.4)

    fig.tight_layout()

    filename = "spatialcnn_error_residuals.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_4()
