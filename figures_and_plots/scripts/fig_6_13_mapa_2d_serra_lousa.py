r"""
Figura 6.13 da Tese:
Continuous 2D spatial comparison over the Serra da Lousa focus area at 25m resolution:
(Left) Digital Elevation Model (SRTM DEM); (Right) Continuous canopy height predictions (0-40 m).

NOTA: Este script requer a biblioteca 'rasterio'.
Recomenda-se executar com o Python do ambiente Conda:
    C:\Users\gui\Documents\Tese\.conda\python.exe fig_6_13_mapa_2d_serra_lousa.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    print("ERRO: 'rasterio' nao esta instalado no Python atual.")
    print("Por favor, execute este script utilizando o ambiente Conda:")
    print(r'  & "C:\Users\gui\Documents\Tese\.conda\python.exe" fig_6_13_mapa_2d_serra_lousa.py')
    sys.exit(1)

# Caminhos base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

OUTPUT_INTERNAL = os.path.join(BASE_DIR, 'outputs', 'SerraLousa_Cortado_2D_Canopy_Height_Map.png')
OUTPUT_THESIS = os.path.abspath(os.path.join(REPO_ROOT, '..', 'Escrita_tese_Rascunho', '5-Figures', 'Results', 'SerraLousa_Cortado_2D_Canopy_Height_Map.png'))

FCH_PATH = os.path.join(REPO_ROOT, 'Deploy_Regional_Model', 'predicted_outputs', 'SerraLousa_Cortado.tif')
DEM_PATH = os.path.join(REPO_ROOT, 'Deploy_Regional_Model', 'aligned_rasters', 'SerraLousa_2020_aligned_42bands.tif')

def gerar_figura():
    if not os.path.exists(FCH_PATH):
        raise FileNotFoundError(f"Raster de predicao FCH nao encontrado: {FCH_PATH}")
    if not os.path.exists(DEM_PATH):
        raise FileNotFoundError(f"Raster DEM nao encontrado: {DEM_PATH}")

    # Leitura do raster FCH cortado (valores em cm, convertidos para metros)
    with rasterio.open(FCH_PATH) as src:
        fch = src.read(1).astype(np.float32) / 100.0
        fch[fch < 0] = np.nan
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Leitura do canal DEM correspondente a mesma janela espacial
    with rasterio.open(DEM_PATH) as src:
        dem_idx = [i + 1 for i, d in enumerate(src.descriptions) if 'DEM' in str(d)][0]
        win = from_bounds(*bounds, transform=src.transform)
        dem = src.read(dem_idx, window=win).astype(np.float32)
        dem[dem < -100] = np.nan

    import matplotlib.ticker as ticker

    # Configuração rigorosamente idêntica à Figura 6.12
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
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

    # Painel Esquerdo: Relevo Topografico (DEM)
    im1 = ax1.imshow(dem, extent=extent, cmap='turbo', vmin=200, vmax=1200, origin='upper')
    ax1.set_title('Terrain Elevation (SRTM DEM)', fontweight='bold', pad=12)
    ax1.set_xlabel('Easting (m, PT-TM06)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Northing (m, PT-TM06)', fontweight='bold', labelpad=10)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5000))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5000))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y):,}'))

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label('Terrain Elevation (m)', fontweight='bold', labelpad=8, fontsize=14)
    cbar1.ax.tick_params(labelsize=13)

    # Painel Direito: Altura de Dossel Continua Prevista (FCH)
    im2 = ax2.imshow(fch, extent=extent, cmap='turbo', vmin=0, vmax=40, origin='upper')
    ax2.set_title('Predicted Forest Canopy Height (FCH)', fontweight='bold', pad=12)
    ax2.set_xlabel('Easting (m, PT-TM06)', fontweight='bold', labelpad=10)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5000))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5000))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{int(y):,}'))
    ax2.tick_params(labelleft=False)  # Oculta rotulos do eixo Y para evitar redundancia

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label('Predicted Canopy Height (m)', fontweight='bold', labelpad=8, fontsize=14)
    cbar2.ax.tick_params(labelsize=13)

    plt.tight_layout()

    # Gravacao dos ficheiros
    os.makedirs(os.path.dirname(OUTPUT_INTERNAL), exist_ok=True)
    plt.savefig(OUTPUT_INTERNAL, dpi=300, bbox_inches='tight')
    print(f"[OK] Gravado: {OUTPUT_INTERNAL}")

    if os.path.exists(os.path.dirname(OUTPUT_THESIS)):
        plt.savefig(OUTPUT_THESIS, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado: {OUTPUT_THESIS}")

    # Atualizar tambem na pasta do Apendice se existir
    appendix_fig = os.path.abspath(os.path.join(REPO_ROOT, '..', 'Escrita_tese_Rascunho', '5-Figures', 'Results', 'comparacao_relevo_altura_lousa_dupla.png'))
    if os.path.exists(os.path.dirname(appendix_fig)):
        plt.savefig(appendix_fig, dpi=300, bbox_inches='tight')
        print(f"[OK] Gravado no apendice: {appendix_fig}")

    plt.close()

if __name__ == '__main__':
    gerar_figura()
