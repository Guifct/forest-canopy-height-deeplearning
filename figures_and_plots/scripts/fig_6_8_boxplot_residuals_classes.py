"""
Figura 6.8 da Tese: Boxplot dos Resíduos por Classe de Altura
Ficheiro de saída: boxplot_residuos_por_classe_altura_aug_on.png

Distribuição estratificada dos resíduos (e = y_pred - y_true) ao longo das 5 classes
de altura (C0 a C4) obtidas por quantis, apresentando para cada classe a linha de mediana
(laranja sólida) e a linha de média (azul tracejada), comprovando a sobrestimação em C0
(+3.50m) e o erro negativo sistemático em C4 (-5.72m).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_fig_6_8(output_dirs=None):
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

    classes = sorted(sub["height_class"].unique())
    data = [sub[sub["height_class"] == c]["residual_m"].to_numpy() for c in classes]

    class_labels = [
        "C0\n(2.4–8.1 m)",
        "C1\n(8.1–12.4 m)",
        "C2\n(12.4–16.7 m)",
        "C3\n(16.7–21.8 m)",
        "C4\n(21.8–63.9 m)"
    ]

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

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    # Boxplot estilizado
    bp = ax.boxplot(
        data,
        tick_labels=class_labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, markerfacecolor='#2c3e50', alpha=0.15, markeredgecolor='none'),
        boxprops=dict(facecolor='#eaf2f8', color='#1a365d', linewidth=1.5),
        whiskerprops=dict(color='#1a365d', linewidth=1.5),
        capprops=dict(color='#1a365d', linewidth=1.5),
        medianprops=dict(color='#d35400', linewidth=2.8),
        meanprops=dict(color='#2980b9', linestyle='--', linewidth=2.5)
    )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Baseline ($e = 0$)')

    # Handles para legenda
    h_zero = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Zero Error Baseline ($e = 0$)')
    h_median = plt.Line2D([0], [0], color='#d35400', linestyle='-', linewidth=2.8, label='Median Residual')
    h_mean = plt.Line2D([0], [0], color='#2980b9', linestyle='--', linewidth=2.5, label='Mean Residual')

    ax.set_xlabel('Stratified Canopy Height Class', fontweight='bold', labelpad=10)
    ax.set_ylabel(r'Residual $e = y_{\mathrm{pred}} - y_{\mathrm{true}}$ (m)', fontweight='bold', labelpad=10)
    ax.set_ylim(-33, 28)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(handles=[h_zero, h_median, h_mean], loc='lower left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#7f8c8d')

    fig.tight_layout()

    filename = "boxplot_residuos_por_classe_altura_aug_on.png"
    for d in output_dirs:
        out_path = d / filename
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_fig_6_8()
