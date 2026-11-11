"""
Figura 6.7 da Tese: Comparação da Distribuição de Alturas Reais vs Previstas (Regressão à Média)
Ficheiro de saída: height_distribution_regression_to_mean.png

Histograma sobreposto de densidade de probabilidade comparando a distribuição
das alturas reais de referência (GEDI RH98, azul) com as previsões do modelo SpatialCNN (vermelho),
evidenciando a contração da variância em torno dos valores médios intermediários (10-22m)
e o estreitamento severo das caudas extremas (<5m e >25m).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_fig_6_7(output_dirs=None):
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
    sub = df[df["augmentation"] == "aug_on"].copy()
    y_true = sub["y_true_m"].to_numpy()
    y_pred = sub["y_pred_m"].to_numpy()

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

    fig, ax = plt.subplots(figsize=(8.8, 5.5))

    bins = np.linspace(0, 45, 46)
    ax.hist(y_true, bins=bins, density=True, alpha=0.60, color='#4a7ab7', edgecolor='#2c5282', linewidth=0.7, label='True Target Distribution (GEDI)')
    ax.hist(y_pred, bins=bins, density=True, alpha=0.55, color='#d9534f', edgecolor='#b52b27', linewidth=0.7, label='SpatialCNN Predictions')

    ax.set_xlabel('Canopy Height (m)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Probability Density', fontweight='bold', labelpad=10)
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 0.085)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#7f8c8d')

    fig.tight_layout()

    filename = "height_distribution_regression_to_mean.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_7()
